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


class ConditionalAffineTransformer(nn.Module):

    def __init__(self, input_dim, condition_dim, split, hidden=512, T=1):
        super().__init__()
        self.split = split
        self.T = T
        # self.constrain = torch.nn.functional.softplus
        self.constrain = torch.exp
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, input_dim),
        )
        self.translation_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, input_dim),
        )

    # masks zeroing (x, y)-input for 2-input NNs seemingly lead to vanishing grads.
    # replacing 2-input 2-output NNs with 1-input/output solved the problem.

    # def forward(self, x, mask=None):
    #     if mask is None:
    #         mask = self.mask
    #     scale = self.scale_net(mask * x)
    #     trans = self.translation_net(mask * x)
    #     y = mask * x + (1 - mask) * (x * torch.exp(scale) + trans)  # affine
    #     return y, scale, trans
    #
    # def inverse(self, y, mask=None):
    #     if mask is None:
    #         mask = self.mask
    #     scale = self.scale_net(mask * y)
    #     trans = self.translation_net(mask * y)
    #     x = mask * y + (1 - mask) * (y - trans) * torch.exp(-scale)  # affine
    #     return x, scale, trans

    def forward(self, x, condition, alternate=False):
        x1 = x[:, :self.split]
        x2 = x[:, self.split:]
        if alternate:
            xc = torch.cat((x1, condition), 1)
            scale = self.constrain(self.scale_net(xc) / self.T)
            trans = self.translation_net(xc)
            y1 = x1
            y2 = x2 * scale + trans
        else:
            xc = torch.cat((x2, condition), 1)
            scale = self.constrain(self.scale_net(xc) / self.T)
            trans = self.translation_net(xc)
            y1 = x1 * scale + trans
            y2 = x2
        y = torch.cat([y1, y2], 1)
        return y, scale, trans

    def inverse(self, y, condition, alternate=False):
        y1 = y[:, :self.split]
        y2 = y[:, self.split:]
        if alternate:
            yc = torch.cat((y1, condition), 1)
            scale = self.constrain(self.scale_net(yc) / self.T)
            trans = self.translation_net(yc)
            x1 = y1
            x2 = (y2 - trans) / scale
        else:
            yc = torch.cat((y2, condition), 1)
            scale = self.constrain(self.scale_net(yc) / self.T)
            trans = self.translation_net(yc)
            x1 = (y1 - trans) / scale
            x2 = y2
        x = torch.cat([x1, x2], 1)
        return x, scale, trans


class ConditionalRealNVP(nn.Module):
    def __init__(self, in_dim, cond_dim, layers=4, hidden=512, T=1, cond_prior=False, uni_prior=False, uni_lim=None, device=None):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        self.input_dim = in_dim
        self.condition_dim = cond_dim
        self.split = 1  # hardcoded for in, out = 1, 1
        self.T=T
        ## masks zeroing (x, y)-input for 2-input NNs seemingly lead to vanishing grads.
        ## replacing 2-input 2-output NNs with 1-input/output solved the problem.
        # self.masks = [
        #     torch.Tensor([1, 0]) if i % 2 == 0 else torch.Tensor([0, 1]) for i in range(0, layers)
        # ]
        self.alternate = [ i % 2 == 0 for i in range(0, layers)
        ]
        # to register submodules params ModuleList is required instead of list
        self.transformers = nn.ModuleList([
            ConditionalAffineTransformer(in_dim, cond_dim, self.split, hidden, T=T)
            for _ in range(0, layers)
        ])
        # self.prior_mean_net = nn.Sequential(
        #     nn.Linear(1, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, 2),
        # )
        # self.prior_scale_net = nn.Sequential(
        #     nn.Linear(1, hidden),
        #     nn.LeakyReLU(),
        #     nn.Linear(hidden, 1),
        # )
        self.cond_prior = cond_prior
        self.noise = None
        self.uni_prior = uni_prior
        self.uni_lim = uni_lim
        if device is not None:
            self.device = device
            self.to(device)

    def cond_base_dist(self, cond):
        if self.uni_prior:
            base_dist = Uniform(-x_lim, x_lim)
        else:
            base_mu, base_cov = torch.zeros(2, device=self.device), torch.eye(2, device=self.device)
            if self.cond_prior:
                # mean = self.prior_mean_net(cond)[0]
                # scale = torch.exp(self.prior_scale_net(cond))[0]
                mean = torch.full_like(base_mu, cond[0, 0])
                scale = 1
                base_mu = base_mu + mean
                base_cov = base_cov * scale
            base_dist = MultivariateNormal(base_mu, base_cov)
        return base_dist

    def forward(self, x, cond):
        log_determinant = 0
        for i in range(0, self.layers):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            x, scale, _ = transformer(x, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        if self.uni_prior:
            log_likelihood = torch.sum(self.cond_base_dist(cond).log_prob(x), dim=1)
        else:
            log_likelihood = self.cond_base_dist(cond).log_prob(x)  # log likelihood of sample under the base measure
        model_loss = - (log_determinant + log_likelihood).mean()
        return x, model_loss

    def inverse(self, batch_size, cond):
        base_dist = self.cond_base_dist(cond)
        # log likelihood of noise under the base measure
        if self.uni_prior:
            y = base_dist.rsample((batch_size, 2))
            log_likelihood = torch.sum(base_dist.log_prob(y), dim=1)
        else:
            y = base_dist.rsample((batch_size,))
            log_likelihood = base_dist.log_prob(y)
        self.noise = y
        log_determinant = 0
        for i in reversed(range(0, self.layers)):
            alternate = self.alternate[i]
            transformer = self.transformers[i]
            y, scale, _ = transformer.inverse(y, cond, alternate)
            log_determinant += torch.log(scale).sum(1)
        model_loss = - (log_determinant + log_likelihood).mean()
        return y, model_loss


def get_batch(batch_size, type, mean=0.0, alpha=1.8, limit=None, device=None):
    rng = np.random.default_rng(seed)
    # ---------- cross
    if type == 0:
        (uni_x1, uni_x2) = rng.uniform(0, 1, size=(2, batch_size))
        x = rng.normal(0, 1, size=batch_size) + uni_x1 * np.cos(2 * pi * uni_x2)
        y = rng.choice([-1, 1], size=batch_size) * x + uni_x1 * np.sin(2 * pi * uni_x2) + mean
        x = x + mean
    # ---------- condensed normal
    if type == 1:
        x = rng.normal(mean, 0.5, size=batch_size)
        y = rng.normal(mean, 0.5, size=batch_size)
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


def get_conditioned_batch(conditions, shuffled=False):
    if shuffled:
        conds = np.random.choice(conditions, size=batch_size)
        cond_batch = [get_batch(1, type=type, mean=c, alpha=alpha, limit=limit, device=device) for c in conds]
        conds = torch.tensor(conds, dtype=torch.float32, device=device).reshape((batch_size, 1))
        return conds, torch.cat(cond_batch, dim=0)
    else:
        cond = np.random.choice(conditions)
        batch = get_batch(batch_size, type=type, mean=cond, alpha=alpha, limit=limit, device=device)
        conds = torch.full_like(batch[:, :1], cond)
        return conds, batch


def calc_true_density(xi_x, xi_y, mean=0, ranges=None):
    xy = np.array(get_batch(50_000, type=type, mean=mean, alpha=alpha, limit=limit))
    p, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=100, range=ranges, density=True)
    p_xi = np.zeros(len(xi_x))
    indx = np.zeros(len(xi_x))
    indy = np.zeros(len(xi_x))
    for i, (x, y) in enumerate(zip(xi_x, xi_y)):
        ind_x = np.argmax(np.where(xedges > x, 1, 0)) - 1
        ind_y = np.argmax(np.where(yedges > y, 1, 0)) - 1
        p_xi[i] = p[ind_x, ind_y]
        indx[i] = ind_x
        indy[i] = ind_y
    return p_xi, indx, indy


# --------------- training_loop
print(f"SEED: {seed}")
batches_number = 10000
batch_size = 1024
shuffle = False
lr = 5e-5
type = 0  # 0 - 4
limit = 5
alpha = 0.5

direct_dens_train = False
x_lim = 4
ranges = [[-x_lim, x_lim], [-x_lim, x_lim]]  # for direct dens training, uniform: [[xmin, xmax], [ymin, ymax]]

#----------------
test_size = 3000
ax_limit = 8
ms = 1
loss_track = []


device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
model = ConditionalRealNVP(in_dim=1, cond_dim=1, layers=8, hidden=1024, T=2, cond_prior=False, uni_prior=False, uni_lim=x_lim, device=device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)
means = np.linspace(-4, 4, 100)

print(f"START ConditionalRealNVP")
print(f"total samples: {batch_size*batches_number}")
sleep(0.01)
t = trange(batches_number, desc='Bar desc', leave=True)
for e in t:
    if direct_dens_train:
        condition = np.random.choice(means)
        conditions = torch.full((batch_size, 1), condition)
        p_nf = model.inverse(batch_size, conditions)[1]
        xi = np.array(model.noise)
        p_xi = calc_true_density(xi[:, 0], xi[:, 1], mean=condition, ranges=ranges)[0]
        p_xi = torch.tensor(p_xi, device=device)
        loss = torch.mean((p_xi - torch.exp(p_nf)) ** 2)
        cond = np.around(float(condition), decimals=5)
    else:
        conditions, batch = get_conditioned_batch(means, shuffle)
        optim.zero_grad()
        x, loss = model(batch, conditions)
        cond = np.around(float(conditions[0, 0]), decimals=5)
    loss_track.append(loss.detach().cpu())
    t.set_description(f"loss = {loss_track[-1]} | cond = {cond}")
    t.refresh()
    loss.backward()
    optim.step()
    # scheduler.step()

print("Inverse pass")
means = [-2, 0.1, 1]
generated = []
priors = []
for mean in means:
    batch = get_batch(test_size, type=type, mean=mean, alpha=alpha, limit=limit, device=device)
    condition = torch.full_like(batch[:, :1], mean, device=device)
    generated.append(model.inverse(test_size, condition)[0].detach().cpu())
    priors.append(model.noise.detach().cpu())


# --------------- visualisation
fig, axs = plt.subplots(1, 3, figsize=(12, 5))
fig.suptitle(f'RealNVP ADAM, layers: {model.layers}, hidden: {model.hidden}, T: {model.T}, seed {seed}\n'
             f'batches_number: {batches_number}, batch: {batch_size}, shuffle: {shuffle}, lr: {lr}, constrain: {model.transformers[0].constrain.__name__}, '
             f'cond_prior: {model.cond_prior}\n direct_dens_train: {direct_dens_train}, uniform: {model.uni_prior}')
for ax, data, mean, prior in zip(axs, generated, means, priors):
    ax.plot(prior[:, 0], prior[:, 1], '.', ms=ms, c='tab:blue', alpha=0.3, label='prior')
    ax.plot(data[:, 0], data[:, 1], '.', ms=ms, c='tab:red', alpha=0.8, label='generated')
    truth = get_batch(test_size, type=type, mean=mean, alpha=alpha, limit=limit)
    ax.plot(truth[:, 0], truth[:, 1], '.', ms=ms, c='tab:green', alpha=0.3, label='truth')
    ax.plot((mean,), (mean,), 'x r')
    ax.set_aspect('equal')
    ax.set_xlim(-ax_limit, ax_limit)
    ax.set_ylim(-ax_limit, ax_limit)
    ax.legend()

fig2, ax2_1 = plt.subplots(1, 1, figsize=(5, 5))
ax2_1.set_title('loss')
ax2_1.plot(loss_track)
ax2_1.set_yscale('log')
ax2_1.set_xlabel('epoch')
ax2_1.set_ylabel('loss')
plt.show()
