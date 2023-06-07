from time import sleep

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from torch.multiprocessing import Manager, Pool, Process, set_start_method
from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg
from aggregation.Static import im_to_pdf, flatten_list, direct_mc, expand_conditions


def prepare_batch_args(process_num, sample_size, dataset, ext_prior, device):
    pool_args = []
    im, prior_im, cond_lst = None, None, None
    for _ in range(process_num):
        # if dataset.name == "dataset_S10_alpha":
        #     x_s, y_s = np.random.choice(x_sources, size=2)
        #     alpha = np.random.choice(alphas)
        #     im = dataset.get_im(x_s, y_s, alpha, data_s)
        #     prior_im = dataset.get_im(x_s, y_s, alpha, prior_s) if ext_prior else None
        #     cond_lst = (x_s, y_s, alpha)
        if dataset.name == "dataset_S10_shape":
            v = np.random.choice(velocity)
            I = np.random.choice(intensity)
            s = np.random.choice(p_size)
            im = dataset.get_im(v, I, s)
            cond_lst = (v, I, s)
        pool_args.append((im, dataset.xedges, cond_lst, sample_size, prior_im, device))
    return pool_args

def get_conditioned_sample(im, xedges, cond_lst, sample_size, prior_im=None, device=None):
    sample = direct_mc(im, xedges, sample_size)
    sample = torch.as_tensor(sample, dtype=torch.float32, device=device)
    conditions = expand_conditions(cond_lst, sample_size, device)
    prior_llk = None
    if prior_im is not None:
        prior_im = np.where(prior_im == 0, 1e-32, prior_im)
        prior_log_pdf = np.log(prior_im / np.sum(prior_im))
        prior_llk = [prior_log_pdf] * sample_size
    return conditions, sample, prior_llk

def get_conditioned_batch(pool, dataset, process_num, sample_size, conds_per_batch=1, ext_prior=False, device=None,):
    conds, batch, base_llk_lst = [], [], []
    runs = int(np.floor(conds_per_batch / process_num))
    residual = conds_per_batch - process_num * runs
    for r in range(runs):
        pool_args = prepare_batch_args(process_num, sample_size, dataset, ext_prior, device)
        results = pool.starmap(get_conditioned_sample, pool_args)
        for result in results:
            conds.append(result[0])
            batch.append(result[1])
            base_llk_lst.append(result[2])
    if residual > 0:
        pool_args = prepare_batch_args(residual, sample_size, dataset, ext_prior, device)
        results = pool.starmap(get_conditioned_sample, pool_args)
        for result in results:
            conds.append(result[0])
            batch.append(result[1])
            base_llk_lst.append(result[2])
    conds = torch.cat(conds, dim=0)
    batch = torch.cat(batch, dim=0)
    base_llk_lst = flatten_list(base_llk_lst) if ext_prior else None
    return conds, batch, base_llk_lst

def train_model(dataset, optim, model, epochs, process_num, samples, conditions_per_batch, external_prior, device=None):
    loss_track = []
    t = trange(epochs, desc='Bar desc', leave=True)
    with Pool(processes=process_num) as pool:
        for e in t:
            conditions, batch, base_llk = get_conditioned_batch(pool, dataset, process_num, samples, conditions_per_batch,
                                                                external_prior, device)
            optim.zero_grad()
            x, loss = model(batch, conditions)
            # x, loss = model(batch, conditions, base_llk, dataset.xedges)
            loss_track.append(np.exp(loss.detach().cpu()))
            t.set_description(f"loss = {loss_track[-1]} |"
                              f" v = {np.around(float(conditions[0, 0]), decimals=5)}"
                              f" I = {np.around(float(conditions[0, 1]), decimals=5)}"
                              f" s = {np.around(float(conditions[0, 2]), decimals=5)}")
            t.refresh()
            loss.backward()
            optim.step()
            # scheduler.step()
    return loss_track

def get_loss_fig(loss_track):
    fig_l, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('loss')
    ax.plot(loss_track)
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    return fig_l

if __name__ == '__main__':
    seed = np.random.randint(10_000)

    # # --------------- dataset_S10_alpha
    # data_name = "dataset_S10_alpha"
    # data_url = "./datasets/dataset_S10_alpha"
    # get_cond_id = lambda x_s, y_s, alpha, n: f"x={x_s}_y={y_s}_angle={int(alpha)}  {int(n)}"
    # data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"
    #
    # external_prior = True
    # x_sources = [(np.around(i * 0.1, decimals=3)) for i in range(1, 10)]
    # alphas = [0, 9, 18, 27, 36, 45, 54, 63, 72, 90]
    # prior_s = 0
    # data_s = 9

    # # --------------- dataset_S10_shape
    data_name = "dataset_S10_shape"
    data_url = "./datasets/dataset_S10_shape"
    get_cond_id = lambda v, I, s: f"v={v} I={int(I)} {int(s)}"
    data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"

    external_prior = False
    p_size = [s for s in range(0, 10)]
    intensity = [1, 10, 50, 100]
    velocity = [0.5, 1.0, 10.0]

    dataset = DatasetImg(data_url, get_cond_id, data_im_name, name=data_name)
    total_conditions = int(len(dataset.x_conditions))
    output = f'./output_2d/{data_name}'

    # --------------- hyperparams & visualization
    batches_number = 1000
    conditions_per_batch = 10
    samples = 1024 * 3
    batch_size = conditions_per_batch * samples
    lr = 1e-7
    test_size = 1000
    ms = 1
    parallel = 10
    reps = 20

    # --------------- model load
    loaded = False
    # loaded = "output_2d/dataset_S10_alpha/30/dataset_S10_alpha_model"
    loaded = "output_2d/shape/4_II/dataset_S10_shape_model"
    print(f"CNF with {parallel} processes")
    print(f"SEED: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        set_start_method('spawn')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'
    except RuntimeError:
        print("failed to use spawn for CUDA")
        device = torch.device('cpu')
        device_name = 'cpu'
    device = torch.device('cpu')
    model = ConditionalRealNVP(cond_dim=3, layers=4, hidden=2024, T=1, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

    # --------------- training_loop
    if loaded:
        model.load_state_dict(torch.load(loaded, map_location=device))
        print(f"loaded model: {loaded}")
    print(f"START ConditionalRealNVP")
    print(f"dataset: {dataset.name} at {dataset.data_loc}")
    print(f"total samples: {batch_size*batches_number}, total conditions: {total_conditions}")
    sleep(0.01)
    loss_track = []
    for r in range(reps):
        lt = train_model(dataset, optim, model, batches_number, parallel, samples, conditions_per_batch, external_prior, device)
        loss_track.append(lt)
        fig_l = get_loss_fig(lt)
        fig_l.savefig(f'{output}_loss_{r}.png')
        np.savetxt(f'{output}_loss_info_{r}.txt', lt)
        torch.save(model.state_dict(), f'{output}_model_{r}')

    torch.save(model.state_dict(), f'{output}_model')
    print("Inverse pass")
    # check_x_s = [0.5, 0.5, 0.8]       # dataset_S10_alpha
    # check_y_s = [0.1, 0.5, 0.5]
    # check_a_s = [18, 45, 63]
    # prior = []
    # generated = []
    # true_pdf_data = []
    # true_pdf_prior = []
    # for x_s, y_s, a_s in zip(check_x_s, check_y_s, check_a_s):
    #     prior_im = dataset.get_im(x_s, y_s, a_s, prior_s)
    #     data_im = dataset.get_im(x_s, y_s, a_s, data_s)
    #     conds = (x_s, y_s, a_s)
    #     conditions, prior_noise, prior_llk = get_conditioned_sample(prior_im, dataset.xedges, conds, test_size, device=device)
    #     inverse_pass = model.inverse_z(prior_noise, conditions, base_llk)[0].detach().cpu()
    #     prior.append(np.array(prior_noise.detach().cpu()))
    #     generated.append(np.array(inverse_pass))
    #     true_pdf_data.append(data_im)
    #     true_pdf_prior.append(prior_im)

    check_v = np.random.choice(velocity, size=3)      # dataset_S10_shape
    check_I = np.random.choice(intensity, size=3)
    check_s = np.random.choice(p_size, size=3)
    check_cond = []
    generated = []
    true_samples = []
    true_pdf_data = []
    for v, I, s in zip(check_v, check_I, check_s):
        conds = (v, I, s)
        data_im = dataset.get_im(v, I, s)
        conditions, data_sample, _ = get_conditioned_sample(data_im, dataset.xedges, conds, test_size, device=device)
        # conditions = expand_conditions(conds, test_size, device)
        inverse_pass = model.inverse(test_size, conditions)[0].detach().cpu()
        generated.append(np.array(inverse_pass))
        true_samples.append(data_sample)
        true_pdf_data.append(data_im)
        check_cond.append(conds)

    # --------------- visualisation
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig_title =  f'Cond RealNVP ADAM, layers: {model.layers}, hidden: {model.hidden}x2, T: {model.T}, seed {seed}\n' \
                 f'lr: {lr}, batches_n: {batches_number}, reps: {reps}, batch: {batch_size}, conds / batch: {conditions_per_batch}\n' \
                 f'constrain: {model.transformers[0].constrain.__name__}, total conditions: {total_conditions}'
    if loaded:
        fig_title += f'\n    loaded model: {loaded}'
    fig.suptitle(fig_title)

    # for ax, sample, im, x_s, y_s in zip(axs[0], prior, true_pdf_prior, check_x_s, check_y_s):
    #     ax.contourf(dataset.xedges, dataset.xedges, im, levels=256, cmap="gist_gray")
    #     ax.plot(sample[:, 1], sample[:, 0], '.', ms=ms, c='tab:blue', alpha=0.8, label='prior')
    #     ax.plot((x_s,), (y_s,), 'x r', label='source')
    #     ax.set_aspect('equal')
    #     ax.axis('off')
    #     # ax.set_xlim(0, 1)
    #     # ax.set_ylim(0, 1)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))
    # for ax, sample, im, x_s, y_s in zip(axs[1], generated, true_pdf_data, check_x_s, check_y_s):
    #     ax.contourf(dataset.xedges, dataset.xedges, im, levels=256, cmap="gist_gray")
    #     ax.plot(sample[:, 1], sample[:, 0], '.', ms=ms, c='tab:green', alpha=0.8, label='generated')
    #     ax.plot((x_s,), (y_s,), 'x r', label='source')
    #     ax.set_aspect('equal')
    #     # ax.set_xlim(0, 1)
    #     # ax.set_ylim(0, 1)
    #     ax.axis('off')
    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))

    for ax, sample, im, true_sample, c in zip(axs, generated, true_pdf_data, true_samples, check_cond):
        ax.set_title(f"v: {c[0]}, I: {c[1]}, s: {c[2]}")
        ax.contourf(dataset.xedges, dataset.xedges, im, levels=256, cmap="gist_gray")
        ax.plot(sample[:, 1], sample[:, 0], '.', ms=ms, c='tab:green', alpha=0.8, label='generated')
        ax.plot(true_sample[:, 1], true_sample[:, 0], '.', ms=ms, c='tab:blue', alpha=0.8, label='true_sample')
        ax.set_aspect('equal')
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        ax.axis('off')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))

    fig2, ax2_1 = plt.subplots(1, 1, figsize=(5, 5))
    ax2_1.set_title('loss')
    ax2_1.plot(np.array(loss_track).flatten())
    ax2_1.set_yscale('log')
    ax2_1.set_xlabel('epoch')
    ax2_1.set_ylabel('loss')

    np.savetxt(f'{output}_loss_info.txt', loss_track, header=fig_title)
    fig.tight_layout()
    fig.savefig(f'{output}_inverse_pass.png', bbox_inches="tight")
    fig2.savefig(f'{output}_loss.png')
    plt.show()

