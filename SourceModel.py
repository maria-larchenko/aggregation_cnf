from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg
from aggregation.Static import expand_conditions

from numpy import cos, sin, pi


class SourceModel:

    def __init__(self, loaded, cond_dim, layers, hidden, T):
        device = torch.device('cpu')
        self.device = device
        self.model = ConditionalRealNVP(cond_dim=cond_dim, layers=layers, hidden=hidden, T=T, device=device, replace_nan=True)
        self.model.load_state_dict(torch.load(loaded, map_location=device))
        self.low = 0
        self.high = 1

    def crop(self, sample):
        sample = np.where(sample < 1, sample, np.nan)
        sample = np.where(0 < sample, sample, np.nan)
        return sample

    def get_sample(self, x_s, y_s, alpha, v, I, s, sample_size=3000):
        # -------- shape_IS
        conds = (I, s)
        conditions = expand_conditions(conds, sample_size, self.device)
        sample, _ = self.model.inverse(sample_size, conditions)
        sample = np.array(sample.detach().cpu())
        # sample = self.crop(sample)
        sample = sample - np.array([0.5, 0])
        # -------- rotate
        a = - alpha * pi / 180
        rot = np.array([[cos(a), -sin(a)],
                          [sin(a), cos(a)]])
        sample = sample.T
        source = np.array([y_s, x_s])
        sample = (rot @ sample).T + source.T
        # -------- crop
        sample = self.crop(sample)
        return sample


loaded = "output_2d/shape_IS/6_II/12/dataset_S10_shape_model"
source_model = SourceModel(loaded, cond_dim=2, layers=6, hidden=512, T=1.27)
x_s, y_s = 0.1, 0.2
alpha = 20
v = 0.5
I = 50.0
s = 9
density = source_model.get_sample(x_s=x_s, y_s=y_s, alpha=alpha, v=1.0, I=I, s=s, sample_size=3000)
plt.plot(density[:, 1], density[:, 0], '.', ms=2, c='green', alpha=1.0, label='generated')
plt.gca().set_title(f'size: {s}, I: {I}')
plt.gca().set_aspect('equal')
plt.gca().set_xlim(-.01, 1.01)
plt.gca().set_ylim(-.01, 1.01)
plt.gca().scatter(x_s, y_s, c='r')
plt.show()
exit()

contour_levels = 5
size = 5

data_name = "dataset_S10_shape"
data_url = "./datasets/dataset_S10_shape"
get_cond_id = lambda v, I, s: f"v={v} I={int(I)} {int(s)}"
data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"

external_prior = False
test_size = 500
p_size = [s for s in range(0, 10)]
intensity = [1, 10, 50, 100]
velocity = [0.5, 1.0, 10.0]

dataset = DatasetImg(data_url, get_cond_id, data_im_name, name=data_name)
xedges = dataset.xedges
check_v = np.random.choice(velocity, size=size)      # dataset_S10_shape
check_I = np.random.choice(intensity, size=size)
check_s = np.random.choice(p_size, size=size)

output = f'./analysis/XsModel'
loaded = "output_2d/shape/4_II/12 20x4000/dataset_S10_shape_model"
device = torch.device('cuda')
model = ConditionalRealNVP(cond_dim=3, layers=6, hidden=512, T=1, device=device, replace_nan=True)
model.load_state_dict(torch.load(loaded, map_location=device))

true_pdfs = []
trained_pdfs = []
generated = []
nan_points = []
inf_points = []
errors = []
check_conds = []
with torch.no_grad():
    for v, I, s in zip(check_v, check_I, check_s):
        conds = (v, I, s)
        check_conds.append(conds)
        im = dataset.get_im(*conds)
        true_pdfs.append(im)

        x = torch.tensor(xedges, dtype=torch.float32, device=device)
        y = torch.tensor(xedges, dtype=torch.float32, device=device)
        # y = torch.tensor(xedges[50:150], dtype=torch.float32, device=device)
        xx = torch.cartesian_prod(x, y)

        conditions = expand_conditions(conds, len(xx), device)

        inverse_cond = expand_conditions(conds, test_size, device)
        inverse_pass = model.inverse(test_size, inverse_cond)[0].detach().cpu()
        generated.append(np.array(inverse_pass))

        x, flatten_log_pdf, nans, infs = model.forward_x(xx, conditions)
        flatten_log_pdf = np.array(flatten_log_pdf.cpu())
        pdf = np.reshape(flatten_log_pdf, im.shape, order='C')
        pdf = np.where(pdf > -30., pdf, -30.)
        pdf = np.exp(pdf)
        trained_pdfs.append(pdf)

        nan_rows = torch.flatten(torch.nonzero(nans[:, 0] + nans[:, 1]))
        broken = xx[nan_rows].cpu() if torch.any(nan_rows) else None
        nan_points.append(broken)

        inf_rows = torch.flatten(torch.nonzero(infs[:, 0] + infs[:, 1]))
        infinite = xx[inf_rows].cpu() if torch.any(inf_rows) else None
        inf_points.append(infinite)
        #
        # err = np.mean(np.abs(im - pdf)/pdf)
        # errors.append(err)

print(f"mean relative error: {np.mean(errors)}")

# --------------- visualisation
fig, axs = plt.subplots(2, size, figsize=(12, 5))
fig.suptitle(f'ACCURACY TEST: {np.around(np.mean(errors), decimals=3)}\n'
             f'Cond RealNVP ADAM, layers: {model.layers}x2, hidden: {model.hidden}, T: {model.T} \n'
             f'{loaded}')

for ax, im, sample, cond in zip(axs[0], true_pdfs, generated, check_conds):
    ax.set_title(f"true p(x), {cond}")
    ax.contourf(xedges, xedges, im, levels=contour_levels, cmap="gist_gray")
    ax.plot(sample[:, 1], sample[:, 0], '.', ms=1, c='green', alpha=1.0, label='generated')
    # ax.plot((x_s,), (y_s,), 'x r', label='source')
    ax.set_aspect('equal')
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    # ax.legend()

for ax, pdf, nan_p, inf_p, cond in zip(axs[1], trained_pdfs, nan_points, inf_points, check_conds):
    ax.set_title(f"model p(x), {cond}")
    ax.contourf(xedges, xedges, pdf, levels=contour_levels, cmap="gist_gray")
    if inf_p is not None:
        ax.plot(inf_p[:, 1], inf_p[:, 0], '.', ms=0.5, c='orange', alpha=1.0, label='produces inf')
    if nan_p is not None:
        ax.plot(nan_p[:, 1], nan_p[:, 0], '.', ms=0.5, c='red', alpha=1.0, label='produces nan')
    # ax.plot((x_s,), (y_s,), 'x r', label='source')
    ax.set_aspect('equal')
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))

fig.tight_layout()
fig.savefig(f"{output}/{datetime.now().strftime('%Y.%m.%d %H-%M-%S')}_test.png", bbox_inches="tight")
plt.show()