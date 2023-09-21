from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from aggregation.CNF import ConditionalRealNVP2D
from aggregation.Dataset import DatasetImg
from aggregation.FFN import FFN
from aggregation.Static import expand_conditions


contour_levels = 5
tests = 5
cond_dim = 2
device = torch.device('cuda')
output = f'./analysis/imshow'

# # ------------------ shape V I S10
# data_name = "dataset_S10_shape"
# data_url = "./datasets/dataset_S10_shape"
# get_cond_id = lambda v, I, s: f"v={v} I={int(I)} {int(s)}"
# data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"
# x_conditions ="__conditions.txt"
# low = 0
# high = 1
#
# external_prior = False
# p_size = [s for s in range(0, 10)]
# intensity = [1, 10, 50, 100]
# velocity = [0.5, 1.0, 10.0]
# get_condition_set = lambda : (np.random.choice(velocity), np.random.choice(intensity), np.random.choice(p_size))

# loaded = "output_2d/shape/4_II/12 20x4000/dataset_S10_shape_model"
# model = ConditionalRealNVP2D(cond_dim=3, layers=4, hidden=2024, T=1, device=device, replace_nan=True)
# loaded = "output_2d/shape/10_II/dataset_S10_shape_model"
# model = ConditionalRealNVP2D(cond_dim=3, layers=10, hidden=2024, T=2, device=device, replace_nan=True)

# # ------------------ shape I S10
# data_name = "dataset_S10_shape"
# data_url = "./datasets/dataset_S10_shape"
# x_conditions ="__conditions_IS.txt"
# get_cond_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
# data_im_name = lambda cond_id: f"{cond_id}_res_100.ppm"
# low = -0.5
# high = 0.5
#
# external_prior = False
# p_size = [s for s in range(0, 10)]
# intensity = [1, 10, 50, 100]
# velocity = [0.5, 1.0, 10.0]
# get_condition_set = lambda : (np.random.choice(intensity), np.random.choice(p_size))
#
# dataset = DatasetImg(data_url, get_cond_id, data_im_name, name=data_name, conditions=x_conditions, low=low, high=high)
# xedges = dataset.xedges
#
# # loaded = "output_2d/shape_IS/6_II/13/dataset_S10_shape_model"  # low=0, high=1
# # model = ConditionalRealNVP2D(cond_dim=cond_dim, layers=6, hidden=512, T=1.27, device=device, replace_nan=True)
# loaded = "output_2d/shape_IS/32/4/dataset_S10_shape_model"   # low=-0.5, high=+0.5
# model = ConditionalRealNVP2D(cond_dim=cond_dim, layers=32, hidden=20, T=1, device=device, replace_nan=True)

# ------------------ shape I S500
data_name = "dataset_S500_shape"
data_url = "./datasets/dataset_S500_shape"
x_conditions ="__conditions_IS.txt"
get_im_id = lambda I, s: f"v=0.5 I={int(I)} {int(s)}"
get_im_name = lambda cond_id: f"{cond_id}_res_1.ppm"
get_concentration_name = lambda I, s: f"v=0.5 I={int(I)} concentration.txt"
get_concentration_scale = lambda I, s: 1.0025 * I
low = -0.5
high = 0.5

p_size = [s for s in range(0, 500)]
intensity = [10*i for i in range(1, 11)]
velocity = [0.5, 1.0, 5.0]
get_condition_set = lambda : (np.random.choice(intensity), np.random.choice(p_size))

dataset = DatasetImg(data_url, get_im_id, get_im_name, get_concentration_name,
                     name=data_name, conditions=x_conditions, low=low, high=high)
xedges = dataset.xedges

loaded_cnf = "output_2d/shape_IS500/32/dataset_S500_shape_model"
model_cnf = ConditionalRealNVP2D(cond_dim=cond_dim, layers=32, hidden=32, T=1, device=device, replace_nan=True)
model_cnf.load_state_dict(torch.load(loaded_cnf, map_location=device))

loaded_ffn = "output_2d/shape_IS500/ffn/dataset_S500_shape_model_ffn"
model_ffn = FFN(cond_dim, 1, hidden=32)
model_ffn.load_state_dict(torch.load(loaded_ffn))

true_pdfs = []
trained_pdfs = []
trained_scale = []
nan_points = []
inf_points = []
errors = []
check_conds = []

with torch.no_grad():
    for _ in range(tests):
        conds = get_condition_set()
        check_conds.append(conds)
        im = dataset.get_im(*conds)
        true_pdfs.append(im)

        ffn_scale = int(model_ffn(conds).detach()[0])
        trained_scale.append(ffn_scale)

        x = torch.tensor(xedges, dtype=torch.float32, device=device)
        y = torch.tensor(xedges, dtype=torch.float32, device=device)
        xx = torch.cartesian_prod(x, y)

        conditions = expand_conditions(conds, len(xx), device)
        # test_size = 500
        # inverse_cond = expand_conditions(conds, test_size, device)
        # inverse_pass = model.inverse(test_size, inverse_cond)[0].detach().cpu()
        # generated.append(np.array(inverse_pass))
        x, flatten_log_pdf, nans, infs = model_cnf.forward_x(xx, conditions)
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

        # err = np.mean(np.abs(im - pdf)/pdf)
        # errors.append(err)

print(f"mean relative error: {np.mean(errors)}")

# --------------- visualisation
fig, axs = plt.subplots(3, tests, figsize=(12, 6))
fig.suptitle(f'CondRealNVP: ADAM, layers: {model_cnf.layers}, hidden: {model_cnf.hidden}x{model_cnf.hidden_layers} {model_cnf.activation_str}, T: {model_cnf.T}\n'
             f'FFN: ADAM, HuberLoss, exp, ReLU hidden: {model_ffn.hidden}x{model_ffn.layers} \n'
             f'cnf: {loaded_cnf} \n'
             f'ffn: {loaded_ffn} \n')

axs[0, 0].set_ylabel("im", fontsize=12) #, rotation=0)
for ax, im, cond in zip(axs[0], true_pdfs, check_conds):
    ax.set_title(f"intensity={cond[0]}, size={cond[1]}")
    ax.imshow(im, cmap='gray', vmin=0, vmax=255)

axs[1, 0].set_ylabel("true p(x)", fontsize=12) #, rotation=0)
for ax, im, cond in zip(axs[1], true_pdfs, check_conds):
    ax.contourf(xedges, xedges, im, levels=contour_levels, cmap="gist_gray")
    ax.text(0.8 * low, 0.8 * high, f'max: {im.max()}', color='w')
    ax.set_xlim(xedges[0]-.01, xedges[-1]+.01)
    ax.set_ylim(xedges[0]-.01, xedges[-1]+.01)

axs[2, 0].set_ylabel("model p(x)", fontsize=12, rotation=90)
for ax, pdf, scale, nan_p, inf_p, cond in zip(axs[2], trained_pdfs, trained_scale, nan_points, inf_points, check_conds):
    ax.contourf(xedges, xedges, pdf, levels=contour_levels, cmap="gist_gray")
    ax.text(0.8 * low, 0.8 * high, f'max: {scale}', color='w')
    if inf_p is not None:
        ax.plot(inf_p[:, 1], inf_p[:, 0], '.', ms=0.5, c='orange', alpha=1.0, label='produces inf')
    if nan_p is not None:
        ax.plot(nan_p[:, 1], nan_p[:, 0], '.', ms=0.5, c='red', alpha=1.0, label='produces nan')
    ax.set_xlim(xedges[0] - .01, xedges[-1] + .01)
    ax.set_ylim(xedges[0] - .01, xedges[-1] + .01)
# ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))

for ax in np.reshape(axs, 3*tests):
    ax.set_aspect('equal')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])



fig.tight_layout()
fig.savefig(f"{output}/{datetime.now().strftime('%Y.%m.%d %H-%M-%S')}_test.png", bbox_inches="tight")
plt.show()