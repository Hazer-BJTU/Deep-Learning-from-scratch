import numpy as np
import torch
from torch.utils import data
from Utility.Synthetic_data import synthetic_data
from torch import nn


def load_array(data_arrays, _batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, _batch_size, shuffle=is_train)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)
print(next(iter(data_iter)))

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, Y in data_iter:
        L = loss(net(X), Y)
        trainer.zero_grad()
        L.backward()
        trainer.step()
    L = loss(net(features), labels)
    print('epoch ', epoch, ', loss ', L)

w = net[0].weight.data
print('true_w - w: ', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('true_b - b: ', true_b - b)
