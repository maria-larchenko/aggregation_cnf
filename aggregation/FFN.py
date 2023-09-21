import numpy as np
import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.constrain = torch.exp
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        self.layers = int(len(self.model) / 2)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32,)
        return torch.exp(self.model(x))