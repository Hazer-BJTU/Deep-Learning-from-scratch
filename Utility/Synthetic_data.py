import numpy as np
import torch
from torch.utils import data


def synthetic_data(_w, _b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(_w)))
    y = torch.matmul(x, _w) + _b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape(-1, 1)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
