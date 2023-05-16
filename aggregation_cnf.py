from time import sleep

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from torch.multiprocessing import Manager, Pool, Process, set_start_method
from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg


def direct_mc(im, xedges, sample_size=1024, prior_im=None):
    scale = int(np.max(im)) + 1
    sample = np.zeros((sample_size, 2))
    base_llk, prior_log_pdf = None, None
    if prior_im is not None:
        base_llk = np.zeros(sample_size)
        prior_im = np.where(prior_im == 0, 1e-32, prior_im)
        prior_log_pdf = np.log(prior_im / np.sum(prior_im))
    i = 0
    while i < sample_size:
        x, y = np.random.uniform(low=0, high=1, size=2)
        ind_x = np.argmax(np.where(xedges > x, 1, 0))
        ind_y = np.argmax(np.where(xedges > y, 1, 0))
        if scale * np.random.uniform() <= im[ind_x, ind_y]:
            sample[i, 0] = x
            sample[i, 1] = y
            if prior_im is not None:
                base_llk[i] = prior_log_pdf[ind_x, ind_y]
            i += 1
    return sample, base_llk

def get_conditioned_sample(im, xedges, conds, sample_size, prior_im=None, device=None):
    sample, base_llk = direct_mc(im, xedges, sample_size, prior_im)
    sample = torch.as_tensor(sample, dtype=torch.float32)
    base_llk = torch.as_tensor(base_llk, dtype=torch.float32) if base_llk is not None else None
    conditions = torch.zeros((sample_size, len(conds)), device=device)
    for i in range(0, len(conds)):
        conditions[:, i] = conds[i]
    if device is not None:
        sample = sample.to(device)
        conditions = conditions.to(device)
        base_llk = base_llk.to(device) if base_llk is not None else None
    return conditions, sample, base_llk

def prepare_batch_args(process_num, sample_size, dataset, cond_prior, device):
    pool_args = []
    for _ in range(process_num):
        x_s, y_s = np.random.choice(x_sources, size=2)
        alpha = np.random.choice(alphas)
        im = dataset.get_im(x_s, y_s, alpha, data_s)
        prior_im = dataset.get_im(x_s, y_s, alpha, prior_s) if cond_prior else None
        pool_args.append((im, dataset.xedges, (x_s, y_s, alpha), sample_size, prior_im, device))
    return pool_args

def get_conditioned_batch(dataset, processes, sample_size, conds_per_batch=1, cond_prior=False, device=None,):
    conds, batch, base_llk = [], [], []
    runs = int(np.floor(conds_per_batch / processes))
    residual = conds_per_batch - processes * runs
    with Pool(processes=processes) as pool:
        for r in range(runs):
            pool_args = prepare_batch_args(processes, sample_size, dataset, cond_prior, device)
            results = pool.starmap(get_conditioned_sample, pool_args)
            for result in results:
                conds.append(result[0])
                batch.append(result[1])
                base_llk.append(result[2])
        if residual > 0:
            pool_args = prepare_batch_args(residual, sample_size, dataset, cond_prior, device)
            results = pool.starmap(get_conditioned_sample, pool_args)
            for result in results:
                conds.append(result[0])
                batch.append(result[1])
                base_llk.append(result[2])
    conds = torch.cat(conds, dim=0)
    batch = torch.cat(batch, dim=0)
    base_llk = torch.cat(base_llk, dim=0) if cond_prior else None
    return conds, batch, base_llk

if __name__ == '__main__':
    seed = 2346  # np.random.randint(10_000)

    # # --------------- dataset
    # data_name = "2D_mono_h=6_D=2_v=1.0_angle=30"
    # data_url = "./datasets/2D_mono_h=6_D=2_v=1.0_angle=30"
    # get_cond_id = lambda x_s, y_s: f"x={x_s}_y={y_s}"
    # data_im_name = lambda cond_id: f"{cond_id}_h=6_D=2_v=1.0_angle=30 0_res_100.ppm"
    #
    # data_name = "dataset_S10 D=1e0"
    # data_url = "./datasets/dataset_S10"
    # get_cond_id = lambda x_s, y_s: f"x={x_s}_y={y_s}"
    # data_im_name = lambda cond_id: f"{cond_id}  0_res_100.ppm"

    data_name = "dataset_S10_alpha"
    data_url = "./datasets/dataset_S10_alpha"
    get_cond_id = lambda x_s, y_s, alpha, n: f"x={x_s}_y={y_s}_angle={int(alpha)}  {int(n)}"
    data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"

    dataset = DatasetImg(data_url, get_cond_id, data_im_name, name=data_name)
    condition_prior = True
    x_sources = [(np.around(i * 0.1, decimals=3)) for i in range(1, 10)]
    alphas = [0, 9, 18, 27, 36, 45, 54, 63, 72, 90]
    prior_s = 0
    data_s = 9
    output = f'./output_2d/{data_name}'
    total_conditions = int(len(dataset.x_conditions) / 2)

    # --------------- hyperparams & visualization
    batches_number = 100
    conditions_per_batch = 20
    samples = 1024 * 2
    batch_size = conditions_per_batch * samples
    lr = 1e-6
    test_size = 3000
    ms = 1
    processes = 10

    # --------------- model load
    loaded = False
    # loaded = "output_2d/dataset_S10 D=1e0/11300/dataset_S10 D=1e0_model"
    print(f"CNF with {processes} processes")
    print(f"SEED: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # try:
    #     set_start_method('spawn')
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     device_name = torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else '-'
    # except RuntimeError:
    #     print("failed to use spawn for CUDA")
    #     device = torch.device('cpu')
    #     device_name = 'cpu'
    device = torch.device('cpu')
    device_name = 'cpu'
    model = ConditionalRealNVP(in_dim=1, cond_dim=3, layers=8, hidden=1024, T=2, external_prior=True, device=device)
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
    t = trange(batches_number, desc='Bar desc', leave=True)
    for e in t:
        conditions, batch, base_llk = get_conditioned_batch(dataset, processes, samples, conditions_per_batch, condition_prior, device)
        optim.zero_grad()
        x, loss = model(batch, conditions, base_llk)
        loss_track.append(np.exp(loss.detach().cpu()))
        t.set_description(f"loss = {loss_track[-1]} |"
                          f" x_s = {np.around(float(conditions[0, 0]), decimals=5)}"
                          f" y_s = {np.around(float(conditions[0, 1]), decimals=5)}")
        t.refresh()
        loss.backward()
        optim.step()
        # scheduler.step()
    torch.save(model.state_dict(), f'{output}_model')

    print("Inverse pass")
    check_x_s = [0.5, 0.5, 0.8]
    check_y_s = [0.1, 0.5, 0.5]
    check_a_s = [18, 45, 63]
    prior = []
    generated = []
    true_pdf_data = []
    true_pdf_prior = []
    for x_s, y_s, a_s in zip(check_x_s, check_y_s, check_a_s):
        prior_im = dataset.get_im(x_s, y_s, a_s, prior_s)
        data_im = dataset.get_im(x_s, y_s, a_s, data_s)
        conds = (x_s, y_s, a_s)
        conditions, prior_noise, base_llk = \
            get_conditioned_sample(prior_im, dataset.xedges, conds, test_size, prior_im=prior_im, device=device)
        inverse_pass = model.inverse_z(prior_noise, conditions, base_llk)[0].detach().cpu()
        prior.append(np.array(prior_noise.detach().cpu()))
        generated.append(np.array(inverse_pass))
        true_pdf_data.append(data_im)
        true_pdf_prior.append(prior_im)

    # --------------- visualisation
    fig, axs = plt.subplots(2, 3, figsize=(12, 5))
    fig_title =  f'Cond RealNVP ADAM, layers: {model.layers}, hidden: {model.hidden}x2, T: {model.T}, seed {seed}\n' \
                 f'lr: {lr}, batches_n: {batches_number}, batch: {batch_size}, conds / batch: {conditions_per_batch}\n' \
                 f'constrain: {model.transformers[0].constrain.__name__}, prior: data s0, total conditions: {total_conditions}'
    if loaded:
        fig_title += f'\n\n loaded model: {loaded}'
    fig.suptitle(fig_title)
    for ax, sample, im, x_s, y_s in zip(axs[0], prior, true_pdf_prior, check_x_s, check_y_s):
        ax.contourf(dataset.xedges, dataset.xedges, im, levels=256, cmap="gist_gray")
        ax.plot(sample[:, 1], sample[:, 0], '.', ms=ms, c='tab:blue', alpha=0.8, label='prior')
        ax.plot((x_s,), (y_s,), 'x r', label='source')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
    for ax, sample, im, x_s, y_s in zip(axs[1], generated, true_pdf_data, check_x_s, check_y_s):
        ax.contourf(dataset.xedges, dataset.xedges, im, levels=256, cmap="gist_gray")
        ax.plot(sample[:, 1], sample[:, 0], '.', ms=ms, c='tab:green', alpha=0.8, label='generated')
        ax.plot((x_s,), (y_s,), 'x r', label='source')
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.legend()

    fig2, ax2_1 = plt.subplots(1, 1, figsize=(5, 5))
    ax2_1.set_title('loss')
    ax2_1.plot(loss_track)
    ax2_1.set_yscale('log')
    ax2_1.set_xlabel('epoch')
    ax2_1.set_ylabel('loss')

    np.savetxt(f'{output}_loss_info.txt', loss_track, header=fig_title)
    fig.savefig(f'{output}_inverse_pass.png')
    fig2.savefig(f'{output}_loss.png')
    plt.show()
