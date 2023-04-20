import imageio.v3 as iio
import imageio.plugins.pyav
import numpy as np
import matplotlib.pyplot as plt

from aggregation.Dataset import read_image

x_s, y_s = 0.5, 0.5
S = 10
cond_id = f"x={x_s}_y={y_s}"
urls = [f"datasets/dataset_S10/{cond_id}_h=6_D=e0_v=1.0_angle=30 {s}_res_100.ppm" for s in range(0, S)]
imgs = [read_image(u) for u in urls]
x_conditions = np.loadtxt("datasets/dataset_S10/_x_conditions.txt")

probes = [
    [130, 126],
    # [230, 70],
    [180, 100],
    [250, 45],
]

colors = [ "tab:blue",
           "tab:green",
           "tab:orange",
           # "tab:red",
           ]

# fig, axs = plt.subplots(2, 5, figsize=(12, 5))
# for i, (ax, im) in enumerate(zip(np.reshape(axs, S), imgs)):
#     ax.set_title(f"c_{i}")
#     ax.imshow(im, origin="lower", cmap="gist_gray")
#     ax.set_aspect("equal")
#     for p, clr in zip(probes, colors):
#         ax.scatter(p[0], p[1], color=clr, marker="x", label=f"probe {p}")
#     ax.axis('off')
# axs[0,2].legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, 1.1))
#
#
# fig2, axs2 = plt.subplots(1, len(probes), figsize=(12, 5))
# for ax, p, clr in zip(axs2, probes, colors):
#     ax.set_title(f"probe {p}")
#     c = []
#     for u, im in zip(urls, imgs):
#         im = im.T
#         c_i = im[p[0], p[1]]
#         c.append(c_i)
#     ax.plot(c, '-o', color=clr, )
#     ax.set_xlabel('particle size')
#
# ## -----------  trying to figure out the mask
# fig3, axs3 = plt.subplots(2, 5, figsize=(12, 5))
# for i, (ax, im) in enumerate(zip(np.reshape(axs3, S), imgs)):
#     if i == 0:
#         ax.set_title(f"c_{i}")
#         ax.imshow(im, origin="lower", cmap="gist_gray")
#         ax.set_aspect("equal")
#         ax.axis('off')
#     if i > 0:
#         ax.set_title(f"c_{i} - c_{0}")
#         diff = im - imgs[0]
#         ax.imshow(diff, origin="lower", cmap="gist_gray")
#         ax.set_aspect("equal")
#         ax.axis('off')

fig4, axs4 = plt.subplots(2, 5, figsize=(14, 5), sharex="all")
a = -0.35
b = 0
c = -0.01
fig4.suptitle(f"Total masses * 10^-6: m = exp({a}x + ln(m_0)) {c}")
for ax in np.reshape(axs4, S):
    x_s, y_s = np.random.choice(x_conditions, size=2)
    ax.set_title(f"x_s: {x_s, y_s}")
    cond_id = f"x={x_s}_y={y_s}"
    urls = [f"datasets/dataset_S10/{cond_id}_h=6_D=e0_v=1.0_angle=30 {s}_res_100.ppm" for s in range(0, S)]
    estimation = []
    total_mass = []
    m = None
    for i, im in enumerate([read_image(u) for u in urls]):
        m = im.sum() / 1_000_000 if m is None else m
        b = np.log(m)
        estimation.append(np.exp(a * i + b) + c)
        total_mass.append(im.sum() / 1_000_000)
    ax.plot(total_mass, "-o")
    ax.plot(estimation, "-")

#
# fig.savefig("c_profiles_1.png",)
# fig2.savefig("c_profiles_2.png",)
# fig3.savefig("c_masks_2.png",)
fig4.savefig("c_total.png", bbox_inches="tight")
plt.show()