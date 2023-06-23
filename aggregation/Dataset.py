import numpy as np
from aggregation.Static import read_image


class DatasetImg:

    def __init__(self, data_loc, get_cond_id, get_im_name, conditions=None, name=None, seed=None):
        self.data_loc = data_loc
        self.get_cond_id = get_cond_id
        self.get_im_name = get_im_name  # is a function of cond_id
        if conditions is not None:
            self.conditions = np.loadtxt(f"{data_loc}/{conditions}", skiprows=1)
        else:
            self.conditions = np.loadtxt(f"{data_loc}/__conditions.txt", skiprows=1)
        self.name = name if name is not None else data_loc
        self.dataset, self.im_size = self._read_dataset()
        self.xedges = np.arange(0, self.im_size) / self.im_size
        self._rng = np.random.default_rng(seed=seed)

    def _read_dataset(self, ):
        dataset = {}
        im = None
        for cond_lst in self.conditions:
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

    def get_random_im(self):
        cond_lst = self._rng.choice(self.conditions)
        return cond_lst, self.get_im(*cond_lst)

