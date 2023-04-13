import torch
import numpy as np
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn


class ModelFNN(nn.Module):

    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, x):
        return self.model(x)


class SinsDataset(Dataset):

    def __init__(self, data_dir, test=False):
        self.test = test
        self.data_dir = data_dir
        self.test_set = set(np.loadtxt(data_dir + 'test_set.txt', dtype=str))
        self.train_set = {file for file in os.listdir(data_dir) if file.endswith('.npy')} - self.test_set
        self.data_set = list(self.test_set if test else self.train_set)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        item = np.load(self.data_dir + self.data_set[idx])
        filename = self.data_set[idx]
        label = filename.replace('omega_', '').replace('.npy', '')
        return item, float(label)


training_data = SinsDataset('datasets/test_sin/')
test_data = SinsDataset('datasets/test_sin/', test=True)

train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

model = ModelFNN(inputs=test_data[0][0].shape[0], hidden=64, outputs=1)
loss_func = nn.MSELoss()
epochs = 50
learning_rate = 0.01

print('TRAIN')
for epoch in range(0, epochs):
    for n, (batch, labels) in enumerate(train_dataloader):

        pred = model(batch.float()).flatten()
        loss = loss_func(pred, labels.float())

        grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
        with torch.no_grad():
            for param, param_grad in zip(model.parameters(), grad):
                param.copy_(param - learning_rate * param_grad)

        if n % 100 == 0:
            print(f'epoch: {epoch}, batch: {n}, loss: {loss}')

print('TEST')
error = []
for batch, labels in test_dataloader:
    with torch.no_grad():
        pred = model(batch.float()).flatten()
        print(f'label freq: {labels[:5]}\npred  freq: {pred[:5]}')
        error.append(torch.abs(pred - labels) / labels)
print(f'MEAN ERROR: {float(torch.mean(torch.tensor(error))) * 100} %')


