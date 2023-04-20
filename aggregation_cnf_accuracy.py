import numpy as np
import numpy.random.mtrand
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange


from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg

seed = 2346  # np.random.randint(10_000)
torch.manual_seed(seed)


# --------------- read_data
data_name = "2D_mono_h=6_D=2_v=1.0_angle=30"
data_url = "./datasets/2D_mono_h=6_D=2_v=1.0_angle=30"
data_im_name = lambda cond_id: f"{cond_id}_h=6_D=2_v=1.0_angle=30 0_res_100.ppm"

data_name = "dataset_S10 D=1e0"
data_url = "./datasets/dataset_S10"
data_im_name = lambda cond_id: f"{cond_id}_h=6_D=e0_v=1.0_angle=30 0_res_100.ppm"
x_conditions = np.loadtxt(data_url + "/_x_conditions.txt")

dataset = DatasetImg(data_url, data_im_name, name=data_name)
im_size = 256
xedges = np.arange(0, im_size) / im_size
contour_levels = 10

# --------------- read_model
device = torch.device('cpu')
replace_nan = True

# path_to_model = "output_2d/2D_mono_h=6_D=2_v=1.0_angle=30/trained/500k, lr=5e-6, batch=1024/2D_mono_h=6_D=2_v=1.0_angle=30_model"
# model_name = "500k, lr=5e-6 batch=1024 * 1, N(0, 1) prior"

path_to_model = "output_2d/dataset_S10 D=1e0/10300/dataset_S10 D=1e0_model"
model_name = "10300, lr: 1e-5, batch: 2048 * 10, N(0.5, 1) prior"

model_name += ", nan -> inf" if replace_nan else ""
model = ConditionalRealNVP(in_dim=1, cond_dim=2, layers=8, hidden=2024, T=2, cond_prior=False, device=device, replace_nan=replace_nan)
model.load_state_dict(torch.load(path_to_model, map_location=device))

# --------------- test loop
check_x_s = [.5]
check_y_s = [.5]
true_pdfs = []
trained_pdfs = []
generated = []
hists = []
errors = []
nan_points = []
inf_points = []

for i in range(0, 20):
    x_s, y_s = np.random.choice(x_conditions, size=2)
    check_x_s.append(x_s)
    check_y_s.append(y_s)

print(f"SEED: {seed}")
with torch.no_grad():
    for i in range(0, len(check_x_s)):
        x_s, y_s = check_x_s[i], check_y_s[i]
        im = dataset.get_im(x_s, y_s)
        im = im / im.sum()
        true_pdfs.append(im)

        dim = len(xedges)
        x = torch.tensor(xedges, dtype=torch.float32, device=device)
        y = torch.tensor(xedges, dtype=torch.float32, device=device)
        xx = torch.cartesian_prod(x, y)
        conditions = torch.zeros((len(xx), 2), device=device)
        conditions[:, 0] = x_s
        conditions[:, 1] = y_s

        x, flatten_log_pdf, nans, infs = model.forward_x(xx, conditions)
        flatten_log_pdf = numpy.array(flatten_log_pdf.cpu())
        pdf = np.reshape(flatten_log_pdf, (dim, dim), order='C')
        pdf = np.where(pdf > -30., pdf, -30.)
        pdf = np.exp(pdf)
        pdf = pdf / pdf.sum()
        trained_pdfs.append(pdf)

        nan_rows = torch.flatten(torch.nonzero(nans[:, 0] + nans[:, 1]))
        broken = xx[nan_rows].cpu() if torch.any(nan_rows) else None
        nan_points.append(broken)
        if broken is not None:
            print(f"({x_s}, {y_s}) {len(broken)} nan points")

        inf_rows = torch.flatten(torch.nonzero(infs[:, 0] + infs[:, 1]))
        infinite = xx[inf_rows].cpu() if torch.any(inf_rows) else None
        inf_points.append(infinite)
        if infinite is not None:
            print(f"({x_s}, {y_s}) {len(infinite)} inf points")

        test_size = 3000
        conditions = torch.zeros((test_size, 2), device=device)
        conditions[:, 0] = x_s
        conditions[:, 1] = y_s
        sample = np.array(model.inverse(test_size, conditions)[0].cpu())
        generated.append(sample)

        ranges = [[0, 1], [0, 1]]
        h, h_xedges, _ = np.histogram2d(sample[:, 0], sample[:, 1], bins=25, range=ranges, density=True)
        h_xedges = h_xedges[:-1]
        hists.append(h)

        err = np.mean(np.abs(im - pdf)/pdf)
        errors.append(err)
        print(f"error ({x_s}, {y_s}) = {err}")

print(f"mean relative error: {np.mean(errors)}")

# --------------- visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle(f'ACCURACY TEST of {len(check_x_s)} imgs = {np.around(np.mean(errors), decimals=3)}\n'
             f'Cond RealNVP ADAM, layers: {model.layers}, hidden: {model.hidden}, T: {model.T}, seed {seed}\n'
             f'{model_name}')
for sample, im, x_s, y_s in zip(generated, true_pdfs, check_x_s, check_y_s):
    ax = ax1
    ax.set_title("true p(x)")
    ax.contourf(xedges, xedges, im, levels=contour_levels, cmap="gist_gray")
    # ax.plot(sample[:, 1], sample[:, 0], '.', ms=1, c='green', alpha=1.0, label='generated')
    ax.plot((x_s,), (y_s,), 'x r', label='source')
    ax.set_aspect('equal')
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    ax.legend()
    break

for pdf, sample, nan_p, inf_p, h, x_s, y_s in zip(trained_pdfs, generated, nan_points, inf_points, hists, check_x_s, check_y_s):
    ax = ax2
    ax.set_title("model p(x)")
    ax.contourf(xedges, xedges, pdf, levels=contour_levels, cmap="gist_gray")
    # ax.plot(sample[:, 1], sample[:, 0], '.', ms=1, c='green', alpha=1.0, label='generated')
    if inf_p is not None:
        ax.plot(inf_p[:, 1], inf_p[:, 0], '.', ms=0.5, c='orange', alpha=1.0, label='produces inf')
    if nan_p is not None:
        ax.plot(nan_p[:, 1], nan_p[:, 0], '.', ms=0.5, c='red', alpha=1.0, label='produces nan')
    # ax.contourf(h_xedges, h_xedges, h, levels=contour_levels, cmap="gist_gray")
    ax.plot((x_s,), (y_s,), 'x r', label='source')
    ax.set_aspect('equal')
    ax.set_xlim(-.01, 1.01)
    ax.set_ylim(-.01, 1.01)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.025))
    break

fig.savefig(f'./{path_to_model}_TEST.png', bbox_inches="tight")
plt.show()
