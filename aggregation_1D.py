import torch
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

seed = 1567 #np.random.randint(10_000)
# random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class ModelFNN(nn.Module):

    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, x):
        return self.model(x)


class MonomersDataset(Dataset):

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
        label = filename.replace('xs_', '').replace('.npy', '')
        # return self.filter_inf(item), int(label)
        return item[:200], int(label)

    def filter_inf(self, item):
        return np.where(np.isinf(item), 7.0, item)


dataset = 'monomers_n2'
training_data = MonomersDataset(f'datasets/{dataset}/')
test_data = MonomersDataset(f'datasets/{dataset}/', test=True)
train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

model = ModelFNN(inputs=test_data[0][0].shape[0], hidden=64, outputs=1)
loss_func = nn.HuberLoss()
epochs = 600
learning_rate = 0.001

print(f'train: {len(training_data)} samples, test: {len(test_data)} samples')
print('TRAIN')
for epoch in range(0, epochs):
    for n, (batch, labels) in enumerate(train_dataloader):

        pred = model(batch.float()).flatten()
        loss = loss_func(pred, labels.float())

        grad = torch.autograd.grad(outputs=loss, inputs=model.parameters())
        with torch.no_grad():
            for param, param_grad in zip(model.parameters(), grad):
                param.copy_(param - learning_rate * param_grad)
    print(f'epoch: {epoch}, loss: {loss}')

print('TEST')
errors = []
labels = []
for batch, label in test_dataloader:
    with torch.no_grad():
        pred = model(batch.float()).flatten()
        print(f'label xs: {label}   pred xs: {pred}')
        errors.append(float(torch.abs(pred - label) / label))
        labels.append(int(label))
print(f'SEED: {seed}')
print(f'MEAN ERROR: {float(np.mean(errors)) * 100} %')

np.savetxt(f'{dataset}_err', [labels, errors])

# \columnwidth 85.29mm - IEEE two-column template, 3.357874 inch
width = 8
plt.rcParams['font.weight'] = 'light'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['savefig.pad_inches'] = 0
plt.rcParams['figure.figsize'] = (width, width/1.9)
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['grid.color'] = 'grey'
plt.rcParams['grid.linewidth'] = 0.1

errors_xs = np.array((labels, errors))
fig, ax = plt.subplots(1, 1)
ax.axhline(5, color='tab:green')
ax.plot(errors_xs[0, :], errors_xs[1, :] * 100, ".")
ax.set_title(f'MEAN ERROR: {float(np.mean(errors)) * 100} %')
ax.set_xlabel('x_s')
ax.set_ylabel('relative error, %')
ax.set_yscale('log')
fig.text(1.0, 0.0, f'SEED: {seed}', ha='right', va='bottom', color='grey')
fig.savefig(f'./{dataset}_err.png')
fig.savefig(f'./{dataset}_err.pdf')
plt.show()
