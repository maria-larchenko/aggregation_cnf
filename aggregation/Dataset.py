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

    def __init__(self, data_loc, get_im_name, get_cond_id=None, x_conditions=None, name=None):
        self.data_loc = data_loc
        self.get_cond_id = lambda x_s, y_s: f"x={x_s}_y={y_s}" if get_cond_id is None else get_cond_id
        self.get_im_name = get_im_name  # is a function of cond_id
        self.x_conditions = np.loadtxt(f"{data_loc}/_x_conditions.txt") if x_conditions is None else x_conditions
        self.name = name if name is not None else data_loc
        self.dataset, self.im_size = self.read_dataset()
        self.xedges = np.arange(0, self.im_size) / self.im_size

    def read_dataset(self,):
        dataset = {}
        for x_s in self.x_conditions:
            for y_s in self.x_conditions:
                cond_id = self.get_cond_id(x_s, y_s)
                im_name = self.get_im_name(cond_id)
                url = f"{self.data_loc}/{im_name}"
                im = read_image(url)
                dataset[cond_id] = im
        im_size = len(im)
        return dataset, im_size

    def get_im(self, x_s, y_s):
        cond_id = self.get_cond_id(x_s, y_s)
        return self.dataset[cond_id]

