from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from aggregation.CNF import ConditionalRealNVP
from aggregation.Dataset import DatasetImg
from aggregation.Static import expand_conditions

from numpy import cos, sin, pi


class SourceModel:

    def __init__(self, loaded, cond_dim, layers, hidden, T):
        device = torch.device('cpu')
        self.device = device
        self.model = ConditionalRealNVP(cond_dim=cond_dim, layers=layers, hidden=hidden, T=T, device=device, replace_nan=True)
        self.model.load_state_dict(torch.load(loaded, map_location=device))
        self.low = 0
        self.high = 1

    def crop(self, sample):
        sample = np.where(sample < self.high, sample, np.nan)
        sample = np.where(self.low < sample, sample, np.nan)
        return sample

    def get_sample(self, x_s, y_s, alpha, v, I, s, sample_size=3000):
        # -------- shape_IS
        conds = (I, s)
        conditions = expand_conditions(conds, sample_size, self.device)
        sample, _ = self.model.inverse(sample_size, conditions)
        sample = np.array(sample.detach().cpu())
        # sample = self.crop(sample)
        sample = sample - np.array([0.5, 0])
        # -------- rotate
        a = - alpha * pi / 180
        rot = np.array([[cos(a), -sin(a)],
                          [sin(a), cos(a)]])
        sample = sample.T
        source = np.array([y_s, x_s])
        sample = (rot @ sample).T + source.T
        # -------- crop
        sample = self.crop(sample)
        return sample


loaded = "../output_2d/shape_IS/6_II/12/dataset_S10_shape_model"
source_model = SourceModel(loaded, cond_dim=2, layers=6, hidden=512, T=1.27)
x_s, y_s = 0.1, 0.2
alpha = 0
v = 0.5
I = 5.0
s = 9
density = source_model.get_sample(x_s=x_s, y_s=y_s, alpha=alpha, v=1.0, I=I, s=s, sample_size=3000)
plt.plot(density[:, 1], density[:, 0], '.', ms=2, c='green', alpha=1.0, label='generated')
plt.gca().set_title(f'size: {s}, I: {I}')
plt.gca().set_aspect('equal')
plt.gca().set_xlim(-.01, 1.01)
plt.gca().set_ylim(-.01, 1.01)
plt.gca().scatter(x_s, y_s, c='r')
plt.show()