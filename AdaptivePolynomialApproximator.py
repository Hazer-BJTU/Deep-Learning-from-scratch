import torch
from torch import nn
import math

import AdvancedCNN_AlexNet
from CIFAR_TEN import load_data_cifar_10
from Utility.GPU import try_gpu
from Fashion_MNIST import load_data_fashion_mnist


def batch_quadratic(X1, A, X2, batch_size, input_features, output_features):
    assert X1.shape == X2.shape and X1.shape[1] == input_features
    assert A.shape[0] == input_features * output_features and A.shape[1] == input_features
    Y = X1 @ A.T
    Y = Y.view(batch_size, output_features, input_features)
    output = torch.bmm(Y, torch.unsqueeze(X2, dim=2))
    return torch.squeeze(output, dim=2)


class AdaptivePolynomialApproximator(nn.Module):
    def __init__(self, features, rank,
                 fact_decay=True, clipping=True, dtype=torch.float32, **kwargs):
        super(AdaptivePolynomialApproximator, self).__init__(**kwargs)
        self.features = features
        self.rank = rank
        self.fact_decay = fact_decay
        self.clipping = clipping

        self.params = nn.Parameter(torch.randn(rank, features * features, features, dtype=dtype))
        self.BN = nn.BatchNorm1d(features)

    def forward(self, X):
        Y_hat = torch.zeros(X.shape[0], self.features, device=X.device)
        Xi = X
        for i in range(self.rank):
            Xi = batch_quadratic(Xi, self.params[i], X, X.shape[0], self.features, self.features)
            Xi = self.BN(Xi)
            if self.fact_decay:
                Y_hat += Xi / math.factorial(i + 1)
            else:
                Y_hat += Xi
        if self.clipping:
            Y_hat = self.BN(Y_hat)
        return Y_hat


class ApaClassifier(nn.Module):
    def __init__(self, input_size, features, categories, rank, hiddens,
                 fact_decay=True, clipping=True, **kwargs):
        super(ApaClassifier, self).__init__(**kwargs)
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, hiddens),
            nn.Tanh(),
            nn.Linear(hiddens, features),
            nn.Tanh()
        )
        self.APA = AdaptivePolynomialApproximator(features, rank, fact_decay, clipping)
        self.drop_out = nn.Dropout(0.5)
        self.linear2 = nn.Sequential(
            nn.Linear(features, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, categories)
        )
        self.residual_link = nn.Sequential(
            nn.Linear(input_size, hiddens),
            nn.Tanh(),
            nn.Linear(hiddens, features),
            nn.Tanh()
        )

    def forward(self, X):
        Y1 = self.drop_out(self.APA(self.linear1(X)))
        Y2 = self.drop_out(self.residual_link(X))
        return self.linear2(Y1 + Y2)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.to(torch.int64) == y)
    return float(cmp.sum().item())


def evaluate(net, data_iter, device):
    net.eval()
    right, total = 0, 0
    with torch.no_grad():
        for i, (X_original, y_original) in enumerate(data_iter):
            X, y = X_original.to(device), y_original.to(device)
            right += accuracy(net(X), y)
            total += X.shape[0]
    return right / total


if __name__ == '__main__':
    lr = 0.001
    num_epoches = 50
    batch_size = 64
    weight_decay = 0.001

    train_iter, test_iter = load_data_cifar_10(batch_size, resize=None)
    device = try_gpu()
    net = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=1, padding='same'),
        nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
        nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
        nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(64),
        nn.Dropout(0.5),
        nn.Flatten(),
        ApaClassifier(1024, 256, 10, 3, 512)
    )
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.85)
    loss = nn.CrossEntropyLoss()

    print(f'training on {device}')
    for epoch in range(num_epoches):
        net.train()
        total_loss, num_samples, train_acc, test_acc = 0, 0, 0, 0
        for i, (X_original, y_original) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X_original.to(device), y_original.to(device)
            y_hat = net(X)
            L = loss(y_hat, y)
            L.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            total_loss += L
            num_samples += X.shape[0]
            with torch.no_grad():
                train_acc += accuracy(y_hat, y)
        print(f'epoce: {epoch}, average loss: {total_loss / num_samples:.3f}, '
              f'average train acc: {train_acc / num_samples:.3f}, '
              f'average valid acc: {evaluate(net, test_iter, device):.3f}')
        scheduler.step()
