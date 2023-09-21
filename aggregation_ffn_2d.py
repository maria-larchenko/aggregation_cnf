from time import sleep

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm, trange
from aggregation.FFN import FFN
from aggregation.Dataset import DatasetImg


def get_loss_fig(loss_track, header):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.set_title(header)
    ax.plot(loss_track)
    ax.set_ylabel('loss')
    ax.set_xlabel('batch')
    ax.set_yscale('log')
    fig.tight_layout()
    return fig

def get_header(ffn_model, dataset):
    header = f'FFN: ADAM, HuberLoss, exp, LeakyReLU hidden: {ffn_model.hidden}x{ffn_model.layers}\n' \
             f'lr: {lr_f}, batches: {batches_number}, batch_size: {batch_size} \n' \
             f'{dataset.name} in [{dataset.low},{dataset.high}], tot.conditions: {total_conditions}, seed {seed}\n'
    if loaded:
        header += f' --- loaded model: {loaded}'
    return header


if __name__ == '__main__':
    seed = np.random.randint(10_000)
    # # --------------- dataset_S500_shape IS
    data_name = "dataset_S500_shape"
    data_url = "./datasets/dataset_S500_shape"
    x_conditions = "__conditions_IS.txt"
    get_im_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
    get_im_name = lambda cond_id: f"{cond_id}_res_1.ppm"
    get_concentration_scale = lambda I, s: 1.0025 * I
    low = -0.5
    high = 0.5

    p_size = [s for s in range(0, 500)]
    intensity = [10 * i for i in range(1, 11)]
    velocity = [0.5, 1.0, 5.0]
    get_condition_set = lambda: (np.random.choice(intensity), np.random.choice(p_size))

    dataset = DatasetImg(data_url, get_im_id, get_im_name, get_concentration_scale,
                         name=data_name, conditions=x_conditions, low=low, high=high)
    cond_dim = int(len(get_condition_set()))
    total_conditions = int(len(dataset.x_conditions))
    output = f'./output_2d/{data_name}'

    # --------------- hyperparams & visualization
    batches_number = 30000
    batch_size = 64
    lr_f = 1e-3

    # --------------- model load
    loaded = False
    loaded = "output_2d/dataset_S500_shape_model_ffn"
    print(f"SEED: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    loss_func = nn.HuberLoss()
    ffn_model = FFN(cond_dim, 1, hidden=32)
    ffn_optim = torch.optim.Adam(ffn_model.parameters(), lr=lr_f)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

    # --------------- training_loop
    print(f"START FFN")
    header = get_header(ffn_model, dataset)
    if loaded:
        ffn_model.load_state_dict(torch.load(loaded))
    print(f"{header}")
    print(f"total samples: {batch_size*batches_number}, total conditions: {total_conditions}")
    sleep(0.01)
    loss_track = []
    t = trange(batches_number, desc='>>> Next train step <<<', leave=True)
    # for b in t:
    #     ffn_model.zero_grad()
    #
    #     batch = [get_condition_set() for _ in range(batch_size)]
    #     im_scales = [dataset.get_im(*cond_set).max() for cond_set in batch]
    #     batch = np.array(batch)
    #     im_scales = torch.as_tensor(im_scales, dtype=torch.float32).flatten()
    #     ffn_scales = ffn_model(batch).flatten()
    #     ffn_loss = loss_func(ffn_scales, im_scales)
    #
    #     loss_track.append(ffn_loss.detach())
    #
    #     t.set_description(f"{b}/{batches_number}|| ffn_loss: {loss_track[-1]}")
    #     t.refresh()
    #     ffn_loss.backward()
    #     ffn_optim.step()

    # --------------- saving_result
    test_size = 30
    batch = [get_condition_set() for _ in range(test_size)]
    im_scales = [dataset.get_im(*cond_set).max() for cond_set in batch]
    batch = np.array(batch)
    ffn_scales = np.array(ffn_model(batch).detach().flatten())
    mean_err = 0
    for im_scale, ffn_scale in zip(im_scales, ffn_scales):
        if im_scale != 0:
            err = np.abs(im_scale - ffn_scale) / ffn_scale
            mean_err += err / test_size
        print(f"true: {im_scale}, learned: {ffn_scale}")
    print(f"mean err: {mean_err}")


    fig_l = get_loss_fig(loss_track, header)
    fig_l.savefig(f'{output}_loss_ffn.png')
    # torch.save(ffn_model.state_dict(), f'{output}_model_ffn')
    print("FINISHED")





