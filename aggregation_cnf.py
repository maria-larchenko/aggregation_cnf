from time import sleep

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from torch.multiprocessing import Manager, Pool, Process, set_start_method
from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg


def direct_mc(im, xedges, sample_size=1024):
    scale = int(np.max(im)) + 1
    sample = np.zeros((sample_size, 2))
    i = 0
    while i < sample_size:
        x, y = np.random.uniform(low=0, high=1, size=2)
        ind_x = np.argmax(np.where(xedges > x, 1, 0))
        ind_y = np.argmax(np.where(xedges > y, 1, 0))
        if np.random.uniform() * scale <= im[ind_x, ind_y]:
            sample[i, 0] = x
            sample[i, 1] = y
            i += 1
    return sample

def get_conditioned_sample(im, xedges, conds, size, device=None):
    x_s, y_s = conds[0], conds[1]
    sample = direct_mc(im, xedges, size)
    sample = torch.as_tensor(sample, dtype=torch.float32)
    cond_x = torch.full_like(sample[:, :1], x_s)
    cond_y = torch.full_like(sample[:, :1], y_s)
    cond = torch.column_stack((cond_x, cond_y))
    if device is not None:
        cond = cond.to(device)
        sample = sample.to(device)
    return cond, sample

def prepare_batch_args(process_num, size, dataset, device):
    pool_args = []
    xedges = dataset.xedges
    for _ in range(process_num):
        x_s, y_s = np.random.choice(dataset.x_conditions, size=2)
        im = dataset.get_im(x_s, y_s)
        pool_args.append((im, xedges, (x_s, y_s), size, device))
    return pool_args

def get_conditioned_batch(dataset, processes, size, conds_per_batch=1, device=None):
    conds, batch = [], []
    runs = int(np.floor(conds_per_batch / processes))
    residual = conds_per_batch - processes * runs
    with Pool(processes=processes) as pool:
        for r in range(runs):
            pool_args = prepare_batch_args(processes, size, dataset, device)
            results = pool.starmap(get_conditioned_sample, pool_args)
            for result in results:
                conds.append(result[0])
                batch.append(result[1])
        if residual > 0:
            pool_args = prepare_batch_args(residual, size, dataset, device)
            results = pool.starmap(get_conditioned_sample, pool_args)
            for result in results:
                conds.append(result[0])
                batch.append(result[1])
    return torch.cat(conds, dim=0), torch.cat(batch, dim=0)

if __name__ == '__main__':
    seed = 2346  # np.random.randint(10_000)

    # --------------- dataset
    data_name = "2D_mono_h=6_D=2_v=1.0_angle=30"
    data_url = "./datasets/2D_mono_h=6_D=2_v=1.0_angle=30"
    data_im_name = lambda cond_id: f"{cond_id}_h=6_D=2_v=1.0_angle=30 0_res_100.ppm"

    data_name = "dataset_S10 D=1e0"
    data_url = "./datasets/dataset_S10"
    data_im_name = lambda cond_id: f"{cond_id}_h=6_D=e0_v=1.0_angle=30 0_res_100.ppm"

    dataset = DatasetImg(data_url, data_im_name, name=data_name)
    x_conditions = dataset.x_conditions
    output = f'./output_2d/{data_name}'

    # --------------- hyperparams & visualization
    batches_number = 100
    conditions_per_batch = 20
    samples = 2048
    batch_size = conditions_per_batch * samples
    lr = 1e-4
    test_size = 3000
    ms = 1
    processes = 8

    # --------------- model load
    loaded = False  # "output_2d/dataset_S10 D=1e0/10300/dataset_S10 D=1e0_model"
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
    model = ConditionalRealNVP(in_dim=1, cond_dim=2, layers=8, hidden=2024, T=2, cond_prior=False, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

    # --------------- training_loop
    if loaded:
        model.load_state_dict(torch.load(loaded, map_location=device))
        print(f"loaded model: {loaded}")
    print(f"START ConditionalRealNVP")
    print(f"dataset: {dataset.name} at {dataset.data_loc}")
    print(f"total samples: {batch_size*batches_number}, total conditions: {len(x_conditions)**2}")
    sleep(0.01)
    loss_track = []
    t = trange(batches_number, desc='Bar desc', leave=True)
    for e in t:
        conditions, batch = get_conditioned_batch(dataset, processes, samples, conditions_per_batch, device)
        optim.zero_grad()
        x, loss = model(batch, conditions)
        loss_track.append(np.exp(loss.detach().cpu()))
        t.set_description(f"loss = {loss_track[-1]} |"
                          f" x_s = {np.around(float(conditions[0, 0]), decimals=5)}"
                          f" y_s = {np.around(float(conditions[0, 1]), decimals=5)}")
        t.refresh()
        loss.backward()
        optim.step()
        # scheduler.step()

    print("Inverse pass")
    check_x_s = [0.5, 0.5, 0.8]
    check_y_s = [0.1, 0.5, 0.5]
    generated = []
    true_pdf = []
    for x_s, y_s in zip(check_x_s, check_y_s):
        conditions = torch.zeros((test_size, 2), device=device)
        conditions[:, 0] = x_s
        conditions[:, 1] = y_s
        inverse_pass = model.inverse(test_size, conditions)[0].detach().cpu()
        generated.append(np.array(inverse_pass))
        true_pdf.append(dataset.get_im(x_s, y_s))

    # --------------- visualisation
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    fig_title =  f'Cond RealNVP ADAM, layers: {model.layers}, hidden: {model.hidden}, T: {model.T}, seed {seed}\n' \
                 f'lr: {lr}, batches_n: {batches_number}, batch: {batch_size}, conds / batch: {conditions_per_batch}\n' \
                 f'constrain: {model.transformers[0].constrain.__name__}, prior: N(0.5, 1), cond_prior: {model.cond_prior}, total conditions: {len(x_conditions)**2}'
    if loaded:
        fig_title += f'\n\n loaded model: {loaded}'
    fig.suptitle(fig_title)
    for ax, sample, im, x_s, y_s in zip(axs, generated, true_pdf, check_x_s, check_y_s):
        ax.contourf(dataset.xedges, dataset.xedges, im, levels=256, cmap="gist_gray")
        ax.plot((x_s,), (y_s,), 'x r', label='source')
        ax.plot(sample[:, 1], sample[:, 0], '.', ms=ms, c='tab:green', alpha=0.8, label='generated')
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    fig2, ax2_1 = plt.subplots(1, 1, figsize=(5, 5))
    ax2_1.set_title('loss')
    ax2_1.plot(loss_track)
    ax2_1.set_yscale('log')
    ax2_1.set_xlabel('epoch')
    ax2_1.set_ylabel('loss')

    torch.save(model.state_dict(), f'{output}_model')
    np.savetxt(f'{output}_conditions.txt', x_conditions)
    np.savetxt(f'{output}_info.txt', x_conditions)
    np.savetxt(f'{output}_loss.txt', loss_track)
    fig.savefig(f'{output}_inverse_pass.png')
    fig2.savefig(f'{output}_loss.png')
    plt.show()
