import imageio.v3 as iio
import imageio.plugins.pyav
import numpy as np
import matplotlib.pyplot as plt

from aggregation.Dataset import read_image

x_s, y_s = 0.5, 0.5
cond_id = f"x={x_s}_y={y_s}"

I = (1, 2, 3, 4, 5, 10)
C = (0, 5, 7)
urls_all = []
for c in C:
    urls_all.append([f"datasets/dataset_S10_intensity/I={intens}_h=6_D=e0_v=1.0_angle=30  {c}_res_100.ppm" for intens in I])
imgs_all = []
for urls in urls_all:
    imgs_all.append([read_image(u) for u in urls])

fig1, axs1 = plt.subplots(len(C), len(I), figsize=(14, 6), sharex="all")
fig1.suptitle("Difference in intensities")
for c, axs, imgs in zip(C, axs1, imgs_all):
    for i, (ax, im, intens) in enumerate(zip(axs, imgs, I)):
        # if i == 0:
        ax.set_title(f"c_{c}, intensity {intens}")
        ax.imshow(im, origin="lower", cmap="gist_gray")
        ax.set_aspect("equal")
        ax.axis('off')
        # if i > 0:
        #     prev = i-1
        #     ax.set_title(f"intensity {intens} - {I[prev]}")
        #     diff = im - imgs[prev]
        #     ax.imshow(diff, origin="lower", cmap="gist_gray")
        #     ax.set_aspect("equal")
        #     ax.axis('off')

fig1.savefig("./analysis/c_intensity_diff.png", bbox_inches="tight")
plt.show()