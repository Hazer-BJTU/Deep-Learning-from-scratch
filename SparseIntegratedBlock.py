import torch
from torch import nn


class BroadenBlock(nn.Module):
    def __init__(self, input_channels, features, factor, **kwargs):
        super(BroadenBlock, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.features = features
        self.factor = factor
        self.block = nn.Linear(features, input_channels * factor * factor)
        nn.init.xavier_uniform_(self.block.weight)

    def forward(self, X):
        batch_size = X.shape[0]
        Y = X.view(batch_size * self.input_channels, self.features)
        Y = self.block(Y).view(batch_size, self.input_channels * self.factor, self.input_channels * self.factor)
        return Y


class IntegratedBlock(nn.Module):
    def __init__(self, input_size, rank, output_channels, output_features, **kwargs):
        super(IntegratedBlock, self).__init__(**kwargs)
        self.input_size = input_size
        self.rank = rank
        self.output_channels = output_channels
        self.output_features = output_features
        self.tanh = nn.Tanh()
        self.params = nn.Parameter(torch.randn(rank, input_size, input_size))
        self.BN = nn.BatchNorm1d((rank + 1) * input_size)
        self.W1 = nn.Parameter(torch.randn(output_channels, (rank + 1) * input_size))
        self.W2 = nn.Parameter(torch.randn(input_size, output_features))
        self.relu = nn.ReLU()

    def forward(self, X):
        Z = self.tanh(X)
        Zi = Z
        output = Zi
        for i in range(self.rank):
            Zi = torch.bmm(torch.matmul(Zi, self.params[i]), Z.mT)
            output = torch.cat((output, Zi), dim=1)
        output = self.BN(output)
        output = torch.matmul(torch.matmul(self.W1, output), self.W2)
        return self.relu(output)


class SparseBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super(SparseBlock, self).__init__(**kwargs)
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(output_channels), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(output_channels), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(output_channels), nn.ReLU(),
            nn.ConvTranspose1d(output_channels, output_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(output_channels), nn.ReLU(),
            nn.ConvTranspose1d(output_channels, output_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(output_channels, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, X):
        return self.block(X)


if __name__ == '__main__':
    net = IntegratedBlock(32, 3, 32, 32)
    print(net.state_dict())
