from time import sleep

import numpy as np
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal
from numpy import pi
from tqdm import tqdm, trange

seed = np.random.randint(10_000)
torch.manual_seed(seed)

# --------------- hyper parameters
batches_number = 150
batch_size = 1024
lr = 1e-6
type = 1  # 0 - 4
limit = 5
alpha = 0.5
shuffle_batch = False
normal_noise = False
x_lim = 4
bins = 100
ranges = [[-x_lim, x_lim], [-x_lim, x_lim]]


class PDFModel(nn.Module):

    def __init__(self, input_dim, condition_dim, hidden, T=1, device=None):
        super().__init__()
        self.T = T
        self.hidden = hidden
        self.log_pdf = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SELU(),
            nn.Linear(hidden, 1),
            # nn.SELU(),
            # nn.Linear(hidden, hidden),
            # nn.SELU(),
            # nn.Linear(hidden, 1),
        )
        self.layers = 5
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x, cond=None):
        # return self.log_pdf(x)
        return torch.exp(self.log_pdf(x) / self.T)

def get_batch(batch_size, type, mean=0.0, alpha=1.8, limit=None, device=None):
    rng = np.random.default_rng(seed)
    # ---------- cross
    if type == 0:
        (uni_x1, uni_x2) = rng.uniform(0, 5, size=(2, batch_size))
        x = rng.normal(0, 1, size=batch_size) + uni_x1 * np.cos(2 * pi * uni_x2)
        y = rng.choice([-1, 1], size=batch_size) * x + uni_x1 * np.sin(2 * pi * uni_x2) + mean
        x = x + mean
    # ---------- condensed normal
    if type == 1:
        x = rng.normal(mean, 3, size=batch_size)
        y = rng.normal(mean, 3, size=batch_size)
    # ---------- cauchy
    if type == 2:
        cauchy_x = rng.standard_cauchy(size=batch_size)
        uni_x = rng.uniform(0, 1, size=batch_size)
        x = cauchy_x * np.cos(2 * pi * uni_x)
        y = cauchy_x * np.sin(2 * pi * uni_x)
    # ---------- limited alpha-stable
    if type == 3:
        angle = rng.uniform(0, 1, size=batch_size)
        radius = np.abs(levy_stable.rvs(alpha, beta=0, size=batch_size))
        radius = np.where(radius < limit, radius, limit)
        x = radius * np.cos(2 * pi * angle)
        y = radius * np.sin(2 * pi * angle)
    # ---------- limited log alpha-stable
    if type == 4:
        angle = rng.uniform(0, 1, size=batch_size)
        radius = np.log(np.abs(levy_stable.rvs(alpha, beta=0, size=batch_size)))
        radius = np.where(radius < limit, radius, limit)
        x = radius * np.cos(2 * pi * angle)
        y = radius * np.sin(2 * pi * angle)
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return torch.column_stack((x, y))

def calc_true_density(mean=0):
    xy = np.array(get_batch(500_000, type=type, mean=mean, alpha=alpha, limit=limit))
    p, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=ranges, density=True)
    return p, xedges, yedges

def get_true_density(xi_x, xi_y, p, xedges, yedges):
    if np.iterable(xi_x):
        p_xi = np.zeros(len(xi_x))
        for i, (x, y) in enumerate(zip(xi_x, xi_y)):
            ind_x = np.argmax(np.where(xedges > x, 1, 0)) - 1
            ind_y = np.argmax(np.where(yedges > y, 1, 0)) - 1
            p_xi[i] = p[ind_x, ind_y]
    else:
        ind_x = np.argmax(np.where(xedges > xi_x, 1, 0)) - 1
        ind_y = np.argmax(np.where(yedges > xi_y, 1, 0)) - 1
        p_xi = p[ind_x, ind_y]
    return p_xi


device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
model = PDFModel(input_dim=2, condition_dim=1, hidden=4048, T=1, device=device)
optim = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)
means = np.linspace(-0, 0, 1)

# --------------- training_loop
print(f"SEED: {seed}")
print(f"START Direct Density")
print(f"total samples: {batch_size*batches_number}")
sleep(0.01)
t = trange(batches_number, desc='Bar desc', leave=True)
loss_track = []
p_true, x_bins, y_bins = calc_true_density(mean=0)
for e in t:
        if normal_noise:
            xi = np.random.normal(0, x_lim, size=(batch_size, 2))
        else:
            xi = np.random.uniform(-x_lim, x_lim, size=(batch_size, 2))
        p = get_true_density(xi[:, 0], xi[:, 1], p_true, x_bins, y_bins)
        p = torch.tensor(p, device=device)
        p_model = model(torch.tensor(xi, dtype=torch.float32))
        loss = torch.mean((p - p_model) ** 2)
        loss_track.append(loss.detach().cpu())
        t.set_description(f"loss = {loss_track[-1]} | cond = {np.around(float(0), decimals=5)}")
        t.refresh()
        loss.backward()
        optim.step()
        # scheduler.step()

# --------------- test
print("Learned density")
p, x, y = calc_true_density(mean=0)
xx, yy = np.meshgrid(x, y)
p_true = np.zeros_like(xx)
p_learned = np.zeros_like(xx)
for i in range(0, len(x)):
    for j in range(0, len(y)):
        xy = torch.tensor((x[i], y[j]), dtype=torch.float32)
        p_learned[i, j] = model(xy).detach().cpu()
        p_true[i, j] = get_true_density(x[i], y[j], p, x, y)

# --------------- visualisation
fig, axs = plt.subplots(2, 3, figsize=(12, 10))
fig.suptitle(f'DirectPDF ADAM, layers: {model.layers}, hidden: {model.hidden}, T: {model.T}, seed {seed}\n'
             f'batches_number: {batches_number}, batch: {batch_size}, lr: {lr}, shuffle_batch: {shuffle_batch}\n'
             f'{ "normal_noise" if normal_noise else "uniform_noise"}')
for ax, zz, mean in zip(axs[0], [p_learned], means):
    ax.contourf(x, y, zz)
    ax.plot((mean,), (mean,), 'x r')
for ax, zz, mean in zip(axs[1], [p_true], means):
    ax.contourf(x, y, zz)
    ax.plot((mean,), (mean,), 'x r')

fig2, ax2_1 = plt.subplots(1, 1, figsize=(5, 5))
ax2_1.set_title('loss')
ax2_1.plot(loss_track)
ax2_1.set_yscale('log')
ax2_1.set_xlabel('epoch')
ax2_1.set_ylabel('loss')
plt.show()
