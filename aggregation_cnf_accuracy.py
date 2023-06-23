from datetime import datetime

import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange


from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg
from aggregation.Static import expand_conditions

seed = np.random.randint(10_000)
torch.manual_seed(seed)


# --------------- read_data
data_name = "2D_mono_h=6_D=2_v=1.0_angle=30"
data_url = "./datasets/2D_mono_h=6_D=2_v=1.0_angle=30"
get_cond_id = lambda x_s, y_s: f"x={x_s}_y={y_s}"
get_im_name = lambda cond_id: f"{cond_id}_h=6_D=2_v=1.0_angle=30 0_res_100.ppm"
x_conditions = "_x_conditions.txt"
cond_dim = 2

data_name = "dataset_S10 D=1e0"
data_url = "./datasets/dataset_S10"
get_cond_id = lambda x_s, y_s: f"x={x_s}_y={y_s}"
get_im_name = lambda cond_id: f"{cond_id}_h=6_D=e0_v=1.0_angle=30 0_res_100.ppm"
x_conditions = "_x_conditions.txt"
cond_dim = 2

data_name = "dataset_S10_shape"
data_url = "./datasets/dataset_S10_shape"
get_cond_id = lambda v, I, s: f"v={v} I={int(I)} {int(s)}"
get_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"
cond_dim = 3

data_name = "dataset_S10_shape_IS"
data_url = "./datasets/dataset_S10_shape"
x_conditions ="__conditions_IS.txt"
get_cond_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
get_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"
cond_dim = 2

dataset = DatasetImg(data_url, get_cond_id, get_im_name, name=data_name, conditions=x_conditions)
xedges = dataset.xedges

# --------------- read_model
loaded = "./output_2d/shape_IS/6_II/12/dataset_S10_shape_model"
output = f'./analysis/'
device = torch.device('cpu')
model = ConditionalRealNVP(cond_dim=2, layers=6, hidden=512, T=1.27, device=device)
model.load_state_dict(torch.load(loaded, map_location=device))

# --------------- run_tests
tests = 5
contour_levels = 5
test_size = 500

check_conds = []
true_pdfs = []
trained_pdfs = []
generated = []
nan_points = []
inf_points = []
errors = []

with torch.no_grad():
    for i in range(tests):
        cond_lst, im = dataset.get_random_im()
        cond_name = f"I={int(cond_lst[0])} s={int(cond_lst[1])}"
        check_conds.append(cond_name)
        true_pdfs.append(im)

        x = torch.tensor(xedges, dtype=torch.float32, device=device)
        y = torch.tensor(xedges, dtype=torch.float32, device=device)
        xx = torch.cartesian_prod(x, y)

        inverse_cond = expand_conditions(cond_lst, test_size, device)
        inverse_pass = model.inverse(test_size, inverse_cond)[0].detach().cpu()
        generated.append(np.array(inverse_pass))

        conditions = expand_conditions(cond_lst, len(xx), device)
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
fig, axs = plt.subplots(2, tests, figsize=(12, 5))
fig.suptitle(f'ACCURACY TEST: {np.around(np.mean(errors), decimals=3)}\n'
             f'Cond RealNVP ADAM, layers: {model.layers}x2, hidden: {model.hidden}, T: {model.T} \n'
             f'{loaded}')

for ax, im, sample, cond in zip(axs[0], true_pdfs, generated, check_conds):
    ax.set_title(f"{cond}")
    ax.contourf(xedges, xedges, im, levels=contour_levels, cmap="gist_gray")
    ax.plot(sample[:, 1], sample[:, 0], '.', ms=1, c='green', alpha=1.0, label='generated')
    # ax.plot((x_s,), (y_s,), 'x r', label='source')
    ax.set_aspect('equal')
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    ax.axis("off")
    # ax.legend()


for ax, pdf, nan_p, inf_p, cond in zip(axs[1], trained_pdfs, nan_points, inf_points, check_conds):
    # ax.set_title(f"model p(x), {cond}")
    ax.contourf(xedges, xedges, pdf, levels=contour_levels, cmap="gist_gray")
    if inf_p is not None:
        ax.plot(inf_p[:, 1], inf_p[:, 0], '.', ms=0.5, c='orange', alpha=1.0, label='produces inf')
    if nan_p is not None:
        ax.plot(nan_p[:, 1], nan_p[:, 0], '.', ms=0.5, c='red', alpha=1.0, label='produces nan')
    # ax.plot((x_s,), (y_s,), 'x r', label='source')
    ax.set_aspect('equal')
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    ax.axis("off")
    # ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))

fig.tight_layout()
fig.savefig(f"{output}/{datetime.now().strftime('%Y.%m.%d %H-%M-%S')}_test.png", bbox_inches="tight")
plt.show()