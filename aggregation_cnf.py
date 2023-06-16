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


# the func depends on a dataset and has to be changed wrt to it
def get_condition_set(dataset):
    # if dataset.name == "dataset_S10_alpha":
    #     x_s, y_s = np.random.choice(x_sources, size=2)
    #     alpha = np.random.choice(alphas)
    #     cond_set = (x_s, y_s, alpha)
    if dataset.name == "dataset_S10_shape":
        # v = np.random.choice(velocity)
        I = np.random.choice(intensity)
        s = np.random.choice(p_size)
        # cond_set = (v, I, s)
        cond_set = (I, s)
    return cond_set

def prepare_batch_args(process_num, sample_size, dataset, ext_prior, device):
    pool_args = []
    im, prior_im, cond_lst = None, None, None
    for _ in range(process_num):
        # if dataset.name == "dataset_S10_alpha":
        #     cond_set = get_condition_set()
        #     im = dataset.get_im(*(cond_set + (data_s)))
        #     prior_im = dataset.get_im(*(cond_set + (prior_s))) if ext_prior else None
        if dataset.name == "dataset_S10_shape":
            cond_set = get_condition_set(dataset)
            im = dataset.get_im(*cond_set)
        pool_args.append((im, dataset.xedges, cond_set, sample_size, prior_im, device))
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

def train_model(rep, pool):
    loss_track = []
    t = trange(batches_number, desc='>>> Next train step <<<', leave=True)
    model.set_temperature(temperatures[rep])
    for b in t:
        conditions, batch, base_llk = get_conditioned_batch(pool, dataset, parallel, samples, conditions_per_batch,
                                                            external_prior, device)
        optim.zero_grad()
        x, model_loss = model(batch, conditions)
        # x, loss = model(batch, conditions, base_llk, dataset.xedges)
        l1reg = 0
        l2reg = 0
        if l1 or l2 is not None:
            for p in model.parameters():
                if l1 is not None:
                    l1reg += torch.abs(p).sum()
                if l2 is not None:
                    l2reg += (p ** 2).sum()
        loss = model_loss + l1 * l1reg + l2 * l2reg
        loss_track.append(loss.detach().cpu())
        t.set_description(f"{rep}/{reps}|| model_loss: {model_loss.detach().cpu()} "
                          f"| l1: {l1 * l1reg} l2: {l2 * l2reg} T: {model.T} | loss: {loss.detach().cpu()}")
        t.refresh()
        loss.backward()
        optim.step()
        # scheduler.step()
    return loss_track

def get_loss_fig(loss_track):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_title('loss')
    ax.plot(loss_track)
    # ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.tight_layout()
    return fig

def get_header(model, dataset):
    header = f'Cond RealNVP ADAM, layers: {model.layers}, hidden: {model.hidden}x2, T: {model.T}, seed {seed}\n' \
                f'lr: {lr}, l1: {l1}, l2: {l2}, batches_n: {batches_number}, reps: {reps}, samples: {samples}, conds / batch: {conditions_per_batch}\n' \
                f'dataset: {dataset.name}, total conditions: {total_conditions} \n'
                # f'constrain: {model.transformers[0].constrain.__name__}, '
    if loaded:
        header += f' --- loaded model: {loaded}'
    return header

def get_inverse_fig(check_num, check_size, model, dataset, title):
    fig, axs = plt.subplots(1, check_num, figsize=(12, 5))
    fig.suptitle(title)
    for ax in axs:
        cond_set = get_condition_set(dataset)
        data_im = dataset.get_im(*cond_set)
        # conditions, data_sample, _ = get_conditioned_sample(data_im, dataset.xedges, cond_set, check_size, device=device)
        conditions = expand_conditions(cond_set, check_size, device)
        inverse_pass = model.inverse(check_size, conditions)[0].detach().cpu()
        ax.contourf(dataset.xedges, dataset.xedges, data_im, levels=256, cmap="gist_gray")
        ax.plot(inverse_pass[:, 1], inverse_pass[:, 0], '.', ms=1, c='tab:green', alpha=0.8, label='inverse_pass')
        # ax.plot(data_sample.cpu()[:, 1], data_sample.cpu()[:, 0], '.', ms=1, c='tab:blue', alpha=0.8, label='true_sample')
        ax.set_aspect('equal')
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        ax.axis('off')
    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))
    fig.tight_layout()
    return fig


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
    x_conditions ="__conditions_IS.txt"
    cond_dim = 2
    # get_cond_id = lambda v, I, s: f"v={v} I={int(I)} {int(s)}"
    get_cond_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
    data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"

    external_prior = False
    p_size = [s for s in range(0, 10)]
    intensity = [1, 10, 50, 100]
    # velocity = [0.5, 1.0, 10.0]

    dataset = DatasetImg(data_url, get_cond_id, data_im_name, name=data_name, conditions=x_conditions)
    total_conditions = int(len(dataset.x_conditions))
    output = f'./output_2d/{data_name}'

    # --------------- hyperparams & visualization
    batches_number = 10000
    conditions_per_batch = 6
    samples = 512 * 3
    batch_size = conditions_per_batch * samples
    lr = 5e-8
    l1 = 1e-4
    l2 = 5e-3
    parallel = 3
    reps = 20
    temperatures = np.linspace(start=1.27, stop=1.27, num=reps)

    # --------------- model load
    loaded = False
    loaded = "output_2d/shape_IS/6_II/12/dataset_S10_shape_model"
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
    # device = torch.device('cpu')
    model = ConditionalRealNVP(cond_dim=cond_dim, layers=6, hidden=512, T=1, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

    # --------------- training_loop
    print(f"START ConditionalRealNVP")
    if loaded:
        model.load_state_dict(torch.load(loaded, map_location=device))
    print(f"{get_header(model, dataset)}")
    print(f"total samples: {batch_size*batches_number}, total conditions: {total_conditions}")
    sleep(0.01)
    loss_track = []
    with Pool(processes=parallel) as pool:
        for r in range(reps):
            lt = train_model(r, pool)
            loss_track.append(lt)

            fig_l = get_loss_fig(np.array(loss_track).flatten())
            fig_l.savefig(f'{output}_loss_{r}.png')
            fig_inv = get_inverse_fig(3, 100, model, dataset, get_header(model, dataset))
            fig_inv.savefig(f'{output}_inverse_pass_{r}.png', bbox_inches="tight")

            np.savetxt(f'{output}_loss_info_{r}.txt', lt)
            torch.save(model.state_dict(), f'{output}_model_{r}')

    # --------------- saving_result
    fig_title = get_header(model, dataset)
    torch.save(model.state_dict(), f'{output}_model')
    np.savetxt(f'{output}_loss_info.txt', loss_track, header=fig_title)
    print("Inverse pass")
    fig_l = get_loss_fig(np.array(loss_track).flatten())
    fig_l.savefig(f'{output}_loss.png')
    fig_inv = get_inverse_fig(5, 3000, model, dataset, fig_title)
    fig_inv.savefig(f'{output}_inverse_pass.png', bbox_inches="tight")
    # plt.show()
    print("FINISHED")





