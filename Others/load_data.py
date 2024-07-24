import scipy
import scipy.io as sio
import torch
from torch.utils import data

sample_rate = 100


def load_data_subject(idx_list, channel, sample_rate, batch_size, oversampling):
    data_array, label_array = None, None
    for idx in idx_list:
        assert 1 <= idx <= 10
        filepath_data = './Data/data/subject' + str(idx) + '.mat'
        filepath_label = './Data/label/' + str(idx) + '-Label.mat'
        data_mat = sio.loadmat(filepath_data)
        label_mat = sio.loadmat(filepath_label)
        if (data_array is None) or (label_array is None):
            data_array = torch.tensor(scipy.signal.resample(data_mat[channel], 30 * sample_rate, axis=1),
                                      dtype=torch.float32)
            label_array = torch.tensor(label_mat['label'], dtype=torch.int64)
            label_array = label_array.view(-1, 1)
        else:
            data_array_append = torch.tensor(scipy.signal.resample(data_mat[channel], 30 * sample_rate, axis=1),
                                             dtype=torch.float32)
            label_array_append = torch.tensor(label_mat['label'], dtype=torch.int64)
            label_array_append = label_array_append.view(-1, 1)
            data_array = torch.cat((data_array, data_array_append), 0)
            label_array = torch.cat((label_array, label_array_append), 0)
    dataset = data.TensorDataset(data_array, label_array)
    if not oversampling:
        return data.DataLoader(dataset, batch_size, shuffle=False)
    weights = []
    for cata in range(5):
        weights.append(label_array.shape[0] / (label_array == cata).sum().item())
    weights_for_samples = []
    for sample in label_array:
        weights_for_samples.append(weights[sample.item()])
    weights_for_samples = torch.tensor(weights_for_samples)
    sampler = data.WeightedRandomSampler(weights_for_samples, label_array.shape[0])
    return data.DataLoader(dataset, batch_size, shuffle=False, sampler=sampler)


def load_data_subject_sequence(idx_list, channel, sample_rate, batch_size, window_size, oversampling):
    data_array, label_array = None, None
    for idx in idx_list:
        assert 1 <= idx <= 10
        filepath_data = './Data/data/subject' + str(idx) + '.mat'
        filepath_label = './Data/label/' + str(idx) + '-Label.mat'
        data_mat = sio.loadmat(filepath_data)
        label_mat = sio.loadmat(filepath_label)
        if (data_array is None) or (label_array is None):
            data_array = torch.tensor(scipy.signal.resample(data_mat[channel], 30 * sample_rate, axis=1),
                                      dtype=torch.float32)
            label_array = torch.tensor(label_mat['label'], dtype=torch.int64)
            label_array = label_array.view(-1, 1)
        else:
            data_array_append = torch.tensor(scipy.signal.resample(data_mat[channel], 30 * sample_rate, axis=1),
                                             dtype=torch.float32)
            label_array_append = torch.tensor(label_mat['label'], dtype=torch.int64)
            label_array_append = label_array_append.view(-1, 1)
            data_array = torch.cat((data_array, data_array_append), 0)
            label_array = torch.cat((label_array, label_array_append), 0)
    data_seqences, label_seqences = None, None
    num_sequences = data_array.shape[0] // window_size
    for seq in range(num_sequences):
        data_seg = data_array[seq * window_size: (seq + 1) * window_size]
        label_seg = label_array[seq * window_size: (seq + 1) * window_size]
        data_seg = torch.unsqueeze(data_seg, dim=0)
        label_seg = torch.unsqueeze(label_seg, dim=0)
        if (data_seqences is None) or (label_seqences is None):
            data_seqences = data_seg.clone()
            label_seqences = label_seg.clone()
        else:
            data_seqences = torch.cat((data_seqences, data_seg), dim=0)
            label_seqences = torch.cat((label_seqences, label_seg), dim=0)
    if oversampling:
        for seq in range(num_sequences):
            bias = window_size // 2
            if (seq + 1) * window_size + bias > data_array.shape[0]:
                break
            data_seg = data_array[seq * window_size + bias: (seq + 1) * window_size + bias]
            label_seg = label_array[seq * window_size + bias: (seq + 1) * window_size + bias]
            data_seg = torch.unsqueeze(data_seg, dim=0)
            label_seg = torch.unsqueeze(label_seg, dim=0)
            data_seqences = torch.cat((data_seqences, data_seg), dim=0)
            label_seqences = torch.cat((label_seqences, label_seg), dim=0)
    dataset = data.TensorDataset(data_seqences, label_seqences)
    return data.DataLoader(dataset, batch_size, shuffle=True)


if __name__ == '__main__':
    train_iter = load_data_subject_sequence([1], 'F3_A2', 100, 16, 10)
    for X, y in train_iter:
        print(X)
        print(y)
        break
