from time import sleep

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from torch import nn
from torch.multiprocessing import Manager, Pool, Process, set_start_method
from aggregation.CNF import ConditionalRealNVP2D
from aggregation.Dataset import DatasetImg
from aggregation.Static import im_to_pdf, flatten_list, direct_mc, expand_conditions


def prepare_batch_args(process_num, sample_size, dataset, ext_prior, device):
    pool_args = []
    im, prior_im, cond_lst = None, None, None
    for _ in range(process_num):
        #     cond_set = get_condition_set()
        #     im = dataset.get_im(*(cond_set + (data_s)))
        #     prior_im = dataset.get_im(*(cond_set + (prior_s))) if ext_prior else None
        cond_set = get_condition_set()
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
    for _ in range(conds_per_batch):
        pool_args = prepare_batch_args(process_num, sample_size, dataset, ext_prior, device)
        results = pool.starmap(get_conditioned_sample, pool_args)
        for result in results:
            conds.append(result[0])
            batch.append(result[1])
            base_llk_lst.append(result[2])
    conds = torch.cat(conds, dim=0)
    batch = torch.cat(batch, dim=0)
    base_llk_lst = flatten_list(base_llk_lst) if ext_prior else None
    return conds, batch, base_llk_lst

def train_model(rep, pool, cnf_model, cnf_optim, dataset):
    loss_track = []
    t = trange(batches_number, desc='>>> Next train step <<<', leave=True)
    cnf_model.set_temperature(temperatures[rep])
    for b in t:
        conditions, batch, base_llk = get_conditioned_batch(pool, dataset, parallel, samples, conditions_per_batch,
                                                            external_prior, device)
        cnf_optim.zero_grad()
        x, cnf_model_loss = cnf_model(batch, conditions)
        # x, loss = model(batch, conditions, base_llk, dataset.xedges)
        l1reg = 0
        l2reg = 0
        if l1 or l2 is not None:
            for p in cnf_model.parameters():
                if l1 is not None:
                    l1reg += torch.abs(p).sum()
                if l2 is not None:
                    l2reg += (p ** 2).sum()
        cnf_loss = cnf_model_loss + l1 * l1reg + l2 * l2reg
        loss_track.append(cnf_loss.detach().cpu())
        # ------- optim part
        t.set_description(f"{rep}/{reps}|| cnf_loss: {cnf_model_loss.detach().cpu()} " 
                          f"| l1: {l1 * l1reg} l2: {l2 * l2reg} T: {cnf_model.T} | total_loss: {loss_track[-1]} ")
        t.refresh()
        cnf_loss.backward()
        cnf_optim.step()
    return loss_track

def get_loss_fig(loss_track):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_title('loss cnf')
    ax1.plot(loss_track)
    ax1.set_xlabel('batch')
    fig.tight_layout()
    return fig

def get_header(cnf_model, dataset):
    header = f'CondRealNVP: ADAM {lr_c}, layers: {cnf_model.layers}, hidden: {cnf_model.hidden}x{cnf_model.hidden_layers} {cnf_model.activation_str}, T: {cnf_model.T}\n' \
             f'lr: {lr_c}, l1: {l1}, l2: {l2}, batches: {batches_number}, samples: {samples}, conds / batch: {conditions_per_batch}, reps: {reps}\n' \
             f'dataset: {dataset.name} in [{dataset.low},{dataset.high}], total conditions: {int(len(dataset.x_conditions))}, seed {seed}\n'
    if loaded:
        header += f' --- loaded model: {loaded}'
    return header

def get_inverse_fig(check_num, check_size, cnf_model, dataset, title):
    fig, axs = plt.subplots(1, check_num, figsize=(12, 5.5))
    fig.suptitle(title)
    for ax in axs:
        cond_set = get_condition_set()
        data_im = dataset.get_im(*cond_set)
        conditions = expand_conditions(cond_set, check_size, device)
        inverse_pass = cnf_model.inverse(check_size, conditions)[0].detach().cpu()
        ax.contourf(dataset.xedges, dataset.xedges, data_im, levels=256, cmap="gist_gray")
        ax.plot(inverse_pass[:, 1], inverse_pass[:, 0], '.', ms=1, c='tab:green', alpha=0.8, label='inverse_pass')
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    seed = np.random.randint(10_000)
    # # --------------- dataset_S10_shape
    # data_name = "dataset_S10_shape"
    # data_url = "./datasets/dataset_S10_shape"
    # x_conditions ="__conditions_IS.txt"
    # cond_dim = 2
    # get_cond_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
    # data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"
    #
    # external_prior = False
    # p_size = [s for s in range(0, 10)]
    # intensity = [1, 10, 50, 100]
    # # velocity = [0.5, 1.0, 10.0]
    # get_condition_set = lambda : (np.random.choice(intensity), np.random.choice(p_size))

    # # --------------- dataset_S500_shape IS
    data_name = "dataset_S500_shape"
    data_url = "./datasets/dataset_S500_shape"
    x_conditions = "__conditions_IS.txt"
    get_im_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
    get_im_name = lambda cond_id: f"{cond_id}_res_1.ppm"
    get_concentration_scale = lambda I, s: 1.0025 * I
    low = -0.5
    high = 0.5
    external_prior = False

    p_size = [s for s in range(0, 500)]
    intensity = [10 * i for i in range(1, 11)]
    velocity = [0.5, 1.0, 5.0]
    get_condition_set = lambda: (np.random.choice(intensity), np.random.choice(p_size))

    dataset = DatasetImg(data_url, get_im_id, get_im_name, get_concentration_scale,
                         name=data_name, conditions=x_conditions, low=low, high=high)
    cond_dim = int(len(get_condition_set()))
    output = f'./output_2d/{data_name}'

    # --------------- hyperparams & visualization
    batches_number = 1000
    conditions_per_batch = 9
    samples = 512 * 2
    batch_size = conditions_per_batch * samples
    lr_c = 1e-4
    l1 = 0 #1e-4
    l2 = 0 #5e-3
    parallel = 9
    reps = 1
    temperatures = np.linspace(start=1.0, stop=1.0, num=reps)

    # --------------- model load
    loaded = False
    # loaded = "output_2d/shape_IS500/32/dataset_S500_shape_model"
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
    cnf_model = ConditionalRealNVP2D(cond_dim=cond_dim, layers=32, hidden=64, T=1, device=device, activation=nn.ReLU)
    cnf_optim = torch.optim.Adam(cnf_model.parameters(), lr=lr_c)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

    # --------------- training_loop
    print(f"START ConditionalRealNVP")
    header = get_header(cnf_model, dataset)
    if loaded:
        cnf_model.load_state_dict(torch.load(loaded, map_location=device))
    print(f"{header}")
    print(f"total samples: {batch_size*batches_number}, total conditions: {int(len(dataset.x_conditions))}")
    sleep(0.01)
    with Pool(processes=parallel) as pool:
        for r in range(reps):
            loss_track = train_model(r, pool, cnf_model, cnf_optim, dataset)

            fig_l = get_loss_fig(loss_track)
            fig_l.savefig(f'{output}_loss_{r}.png')
            fig_inv = get_inverse_fig(5, 1000, cnf_model, dataset, header)
            fig_inv.savefig(f'{output}_inverse_pass_{r}.png', bbox_inches="tight")

            np.savetxt(f'{output}_loss_info_{r}.txt', loss_track)
            torch.save(cnf_model.state_dict(), f'{output}_model_{r}')

    # --------------- saving_result
    torch.save(cnf_model.state_dict(), f'{output}_model')

    print("Inverse pass")
    fig_inv = get_inverse_fig(5, 3000, cnf_model, dataset, header)
    fig_inv.savefig(f'{output}_inverse_pass.png', bbox_inches="tight")
    print("FINISHED")





