import numpy as np
from aggregation.Static import read_image
from typing import Callable



class DatasetImg:

    def __init__(self, data_loc: str,
                 get_im_id: Callable,
                 get_im_name: Callable,
                 get_concentration_scale: Callable = None,
                 get_concentration_name: Callable = None,
                 conditions = None,
                 name = None,
                 low = 0,
                 high = 1.0):
        self.data_loc = data_loc
        self.get_im_id = get_im_id
        self.get_im_name = get_im_name  # is a function of get_im_id
        self.get_concentration_scale = get_concentration_scale  # calculates scale, if provided
        self.get_concentration_name = get_concentration_name  # reads scale from a text file, if provided, can cause exception
        self.low = low
        self.high = high
        if conditions is not None:
            self.x_conditions = np.loadtxt(f"{data_loc}/{conditions}", skiprows=1)
        else:
            self.x_conditions = np.loadtxt(f"{data_loc}/__conditions.txt", skiprows=1)
        self.name = name if name is not None else data_loc
        self.dataset, self.im_size = self.read_dataset()
        self.xedges = np.linspace(low, high, self.im_size)
        self.concentrations = {}

    def read_dataset(self,):
        dataset = {}
        for cond_lst in self.x_conditions:
            im_id = self.get_im_id(*cond_lst)
            im_name = self.get_im_name(im_id)
            url = f"{self.data_loc}/{im_name}"
            im = read_image(url)
            dataset[im_id] = im
        im_size = len(im)
        return dataset, im_size

    def get_im(self, *cond_lst):
        im_id = self.get_im_id(*cond_lst)
        return self.dataset[im_id]

    def get_concentration(self, *cond_lst):
        if self.get_concentration_scale is not None:
            return self.get_concentration_scale(*cond_lst)
        if self.get_concentration_name is not None:
            c_name = self.get_concentration_name(*cond_lst)
            if c_name in self.concentrations.keys():
                return self.concentrations[c_name]
            c = np.loadtxt(f"{self.data_loc}/{c_name}")   # can cause exception
            self.concentrations[c_name] = c
            return c
        return 1
