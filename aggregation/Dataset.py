import numpy as np
from aggregation.Static import read_image



class DatasetImg:

    def __init__(self, data_loc, get_cond_id, get_im_name, conditions=None, name=None):
        self.data_loc = data_loc
        self.get_cond_id = get_cond_id
        self.get_im_name = get_im_name  # is a function of cond_id
        if conditions is not None:
            self.x_conditions = np.loadtxt(f"{data_loc}/{conditions}", skiprows=1)
        else:
            self.x_conditions = np.loadtxt(f"{data_loc}/__conditions.txt", skiprows=1)
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

