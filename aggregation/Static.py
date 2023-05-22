import numpy as np
import imageio.v3 as iio
import imageio.plugins.pyav
from numba import jit, prange


@jit(parallel=True)
# @jit(nopython=True, parallel=True)
def external_prior_llk(x, base_llk_lst, xedges):
    """assuming x and base_llk as list with ref to 2D ndarrays"""
    n = len(base_llk_lst)
    p = np.zeros(n)
    for i in prange(n):
        llk = base_llk_lst[i]
        x_i = x[i]
        p[i] = im_to_pdf(llk, xedges, x_i[0], x_i[1])
    return p

def flatten_list(l):
    return [item for sublist in l for item in sublist]

@jit(nopython=True)
def im_to_pdf(im, xedges, x, y, low=0, high=1):
    if low < x < high and low < y < high:
        ind_x = np.argmax(np.where(xedges > x, 1, 0)) - 1
        ind_y = np.argmax(np.where(xedges > y, 1, 0)) - 1
        return im[ind_x, ind_y]
    return 0


def read_image(url):
    im = iio.imread(url)
    im = im[0]
    im = im[:, :, 0]
    # im = np.mean(im[0], axis=2)
    # integral = im.sum()
    # im = im / integral
    return im

