from time import sleep

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange

from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg

seed = 2346  # np.random.randint(10_000)
torch.manual_seed(seed)


# --------------- dataset
data_name = "2D_mono_h=6_D=2_v=1.0_angle=30"
data_url = "./datasets/2D_mono_h=6_D=2_v=1.0_angle=30"
data_im_name = lambda cond_id: f"{cond_id}_h=6_D=2_v=1.0_angle=30 0_res_100.ppm"

data_name = "dataset_S10 D=1e0"
data_url = "./datasets/dataset_S10"
data_im_name = lambda cond_id: f"{cond_id}_h=6_D=e0_v=1.0_angle=30 5_res_100.ppm"

dataset = DatasetImg(data_url, data_im_name, name=data_name)
x_conditions = dataset.x_conditions
im_size = 256
xedges = np.arange(0, im_size) / im_size
output = f'./output_2d/{data_name}'

# --------------- hyperparams & visualization
batches_number = 100
conditions_per_batch = 10
samples = 2048
batch_size = conditions_per_batch * samples
lr = 1e-4
test_size = 3000
ms = 1

# --------------- model load
loaded = False # "output_2d/trained_3/5300/2D_mono_h=6_D=2_v=1.0_angle=30_model"
device = torch.device('cpu')
model = ConditionalRealNVP(in_dim=1, cond_dim=2, layers=8, hidden=2024, T=2, cond_prior=False, device=device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)


def direct_mc(im, sample_size=1024):
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

def get_conditioned_batch(dataset, size, device=None):
    x_s, y_s = np.random.choice(x_conditions, size=2)
    im = dataset.get_im(x_s, y_s)
    sample = direct_mc(im, size)
    batch = torch.as_tensor(sample, dtype=torch.float32)
    cond_x = torch.full_like(batch[:, :1], x_s)
    cond_y = torch.full_like(batch[:, :1], y_s)
    cond = torch.column_stack((cond_x, cond_y))
    if device is not None:
        cond = cond.to(device)
        batch = batch.to(device)
    return cond, batch

def get_conditioned_batch_2(dataset, samples, conditions_per_batch=1, device=None):
    if conditions_per_batch > 1:
        cond, batch = [], []
        for i in range(0, conditions_per_batch):
            c, b = get_conditioned_batch(dataset, samples, device)
            cond.append(c)
            batch.append(b)
        return torch.cat(cond, dim=0), torch.cat(batch, dim=0)
    else:
        return get_conditioned_batch(dataset, samples, device)


# --------------- training_loop
print(f"SEED: {seed}")
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
    conditions, batch = get_conditioned_batch_2(dataset, samples, conditions_per_batch, device)
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
check_x_s = [0.1, 0.5, 0.8]
check_y_s = [0.1, 0.1, 0.5]
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
    ax.contourf(xedges, xedges, im, levels=256, cmap="gist_gray")
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
