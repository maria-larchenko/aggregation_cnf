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

fig, axs = plt.subplots(2, 5, figsize=(12, 5))
for ax, im in zip(np.reshape(axs, S), imgs):
    ax.imshow(im, origin="lower", cmap="gist_gray")
    ax.set_aspect("equal")
    for p, clr in zip(probes, colors):
        ax.scatter(p[0], p[1], color=clr, marker="x", label=f"probe {p}")
    ax.axis('off')
axs[0,2].legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, 1.1))


fig2, axs2 = plt.subplots(1, len(probes), figsize=(12, 5))
for ax, p, clr in zip(axs2, probes, colors):
    ax.set_title(f"probe {p}")
    c = []
    for u, im in zip(urls, imgs):
        im = im.T
        c_i = im[p[0], p[1]]
        c.append(c_i)
    ax.plot(c, '-o', color=clr, )
    ax.set_xlabel('particle size')

fig.savefig("c_profiles_1.png",)
fig2.savefig("c_profiles_2.png",)
plt.show()