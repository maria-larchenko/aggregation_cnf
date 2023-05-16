import numpy as np
import imageio.v3 as iio
import imageio.plugins.pyav


def read_image(url):
    im = iio.imread(url)
    im = im[0]
    im = im[:, :, 0]
    # im = np.mean(im[0], axis=2)
    # integral = im.sum()
    # im = im / integral
    return im


class DatasetImg:

    def __init__(self, data_loc, get_cond_id, get_im_name, conditions=None, name=None):
        self.data_loc = data_loc
        self.get_cond_id = get_cond_id
        self.get_im_name = get_im_name  # is a function of cond_id
        self.x_conditions = np.loadtxt(f"{data_loc}/__conditions.txt", skiprows=1) if conditions is None else conditions
        self.name = name if name is not None else data_loc
        self.dataset, self.im_size = self.read_dataset()
        self.xedges = np.arange(0, self.im_size) / self.im_size

    def read_dataset(self,):
        dataset = {}
        for cond_lst in self.x_conditions:
            cond_id = self.get_cond_id(*cond_lst)
            im_name = self.get_im_name(cond_id)
            url = f"{self.data_loc}/{im_name}"
            im = read_image(url)
            dataset[cond_id] = im
        im_size = len(im)
        return dataset, im_size

    def get_im(self, *cond_lst):
        cond_id = self.get_cond_id(*cond_lst)
        return self.dataset[cond_id]

