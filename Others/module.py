import torch
from torch import nn
from load_data import load_data_subject

sample_rate = 100
batch_size = 16


def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)


def get_conv_net(sample_rate, conv_type):
    assert conv_type == 1 or conv_type == 2
    kernel_size, stride = 0, 0
    size_subsequent = (1, 1, 1)
    if conv_type == 1:
        kernel_size = sample_rate // 2
        stride = sample_rate // 16
        size_subsequent = (8, 8, 4)
    elif conv_type == 2:
        kernel_size = sample_rate * 4
        stride = sample_rate // 2
        size_subsequent = (4, 6, 2)
    conv_net = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=kernel_size, stride=stride,
                  padding=kernel_size // 2, bias=False),
        nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(kernel_size=size_subsequent[0], stride=size_subsequent[0]),
        nn.Dropout(0.5),
        nn.Conv1d(64, 128, kernel_size=size_subsequent[1], stride=1,
                  padding=size_subsequent[1] // 2, bias=False),
        nn.BatchNorm1d(128), nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=size_subsequent[1], stride=1,
                  padding=size_subsequent[1] // 2, bias=False),
        nn.BatchNorm1d(128), nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=size_subsequent[1], stride=1,
                  padding=size_subsequent[1] // 2, bias=False),
        nn.BatchNorm1d(128), nn.ReLU(),
        nn.MaxPool1d(kernel_size=size_subsequent[2], stride=size_subsequent[2]),
        nn.Dropout(0.5)
    )
    conv_net.apply(init_weight)
    return conv_net


class FeatureExtraction(nn.Module):
    def __init__(self, sample_rate, **kwargs):
        super(FeatureExtraction, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.conv1 = get_conv_net(sample_rate, 1)
        self.conv2 = get_conv_net(sample_rate, 2)

    def forward(self, X):
        F1 = self.conv1(X)
        F2 = self.conv2(X)
        F = torch.cat((F1, F2), dim=2)
        return F


def get_rl_net(sample_rate, input_nums, output_nums):
    hidden_nums = int((input_nums + output_nums) * 1.28)
    rl_net = nn.Sequential(
        FeatureExtraction(sample_rate),
        nn.Flatten(),
        nn.Linear(input_nums, hidden_nums),
        nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(hidden_nums, hidden_nums),
        nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(hidden_nums, output_nums)
    )
    rl_net.apply(init_weight)
    return rl_net


class SequenceLearning(nn.Module):
    def __init__(self, pretrain_path, sample_rate, input_size, device, **kwargs):
        super(SequenceLearning, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.input_size = input_size
        self.num_layers = 2
        self.rnn_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=1024,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=False,
        )
        self.residual_link = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 5)
        )
        self.residual_link.apply(init_weight)
        self.classifier.apply(init_weight)
        self.rnn_layer.to(device)
        self.residual_link.to(device)
        self.classifier.to(device)
        self.feature_extraction = torch.load(pretrain_path, map_location=device)

    def get_initial_states(self, batch_size, device):
        H0 = torch.zeros((self.num_layers, batch_size, 1024), device=device)
        C0 = torch.zeros((self.num_layers, batch_size, 1024), device=device)
        return H0, C0

    def forward(self, X, in_states, window_size):
        A = self.feature_extraction(X)
        A = A.view(A.shape[0], self.input_size)
        FC_out = self.residual_link(A)
        lstm_out, out_states = self.rnn_layer(A.view(A.shape[0] // window_size, window_size, A.shape[1]),
                                              in_states)
        lstm_out = lstm_out.reshape((A.shape[0], 1024))
        Y = FC_out + lstm_out
        Z = self.classifier(Y)
        return Z, out_states


if __name__ == '__main__':
    data_iter = load_data_subject([1], 'F3_A2', sample_rate, batch_size, True)
    net = get_rl_net(sample_rate, input_nums=25 * 128, output_nums=5)
    for X, y in data_iter:
        net.train()
        X = torch.unsqueeze(X, dim=1)
        y_hat = net(X)
        print(y_hat.shape)
        break
