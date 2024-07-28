import torch
import math
import argparse
import sys
import copy

from torch import nn
from CIFAR_TEN import load_data_cifar_10
from Utility.GPU import try_gpu
from AdaptivePolynomialApproximator import AdaptivePolynomialApproximator
from AdaptivePolynomialApproximator import batch_quadratic


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class FeatureExtraction(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtraction, self).__init__(**kwargs)
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding='same'),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding='same'),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.drop_out = nn.Dropout(0.5)

    def forward(self, X):
        p1 = self.conv1_1(X)
        f1 = self.pooling(self.conv1_2(p1) + p1)
        p2 = self.conv2_1(X)
        f2 = self.pooling(self.conv2_2(p2) + p2)
        features = torch.cat((f1, f2), dim=1)
        return self.drop_out(features)


class ApaBlock(nn.Module):
    def __init__(self, input_size, hiddens, rank, **kwargs):
        super(ApaBlock, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.rank = rank
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, hiddens),
            nn.Tanh()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_size, hiddens),
            nn.Tanh()
        )
        self.params = nn.Parameter(torch.randn(rank, hiddens * hiddens, hiddens))
        self.BNZ = nn.BatchNorm1d(hiddens)
        self.BNY = nn.BatchNorm1d(hiddens)
        self.drop_out = nn.Dropout(0.5)

    def forward(self, X):
        Z = self.linear1(X)
        Y = self.linear2(X)
        Zi = Z
        for i in range(self.rank):
            Zi = batch_quadratic(Zi, self.params[i], Z, X.shape[0], self.hiddens, self.hiddens)
            Zi = self.BNZ(Zi)
            Y = Y + Zi
        Y = self.BNY(Y)
        return self.drop_out(Y)


class ApaFeatureNet(nn.Module):
    def __init__(self, **kwargs):
        super(ApaFeatureNet, self).__init__(**kwargs)
        self.F = FeatureExtraction()
        self.APA1 = ApaBlock(1024, 196, 5)
        self.APA2 = ApaBlock(1024, 196, 5)
        self.APA3 = ApaBlock(1024, 196, 5)
        self.dense = nn.Sequential(
            nn.Linear(588, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )

    def forward(self, X):
        f = self.F(X)
        apaout1 = self.APA1(f)
        apaout2 = self.APA2(f)
        apaout3 = self.APA3(f)
        apaout = torch.cat((apaout1, apaout2, apaout3), dim=1)
        return self.dense(apaout)


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
    net = ApaFeatureNet()
    net.apply(init_weights)

    parser = argparse.ArgumentParser(description='Choose device')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0)
    parser.add_argument('--num_epochs', type=int, nargs='?', default=160)
    parser.add_argument('--lr', type=float, nargs='?', default=0.001)
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0001)
    parser.add_argument('--batch_size', type=int, nargs='?', default=256)
    args = parser.parse_args()

    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    weight_decay = args.weight_decay

    train_iter, test_iter = load_data_cifar_10(batch_size, resize=None)
    device = try_gpu(args.cuda_idx)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.85)
    loss = nn.CrossEntropyLoss()

    best_net, best_valid_acc, info = None, 0.0, []
    print(f'training on {device}')
    for epoch in range(num_epochs):
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
        valid_acc = evaluate(net, test_iter, device)
        print(f'epoce: {epoch}, average loss: {total_loss / num_samples:.3f}, '
              f'average train acc: {train_acc / num_samples:.3f}, '
              f'average valid acc: {valid_acc:.3f}')
        if valid_acc > best_valid_acc and epoch + 1 > int(0.5 * num_epochs):
            best_valid_acc = valid_acc
            best_net = copy.deepcopy(net)
            info.append((total_loss / num_samples, train_acc / num_samples, valid_acc))
        scheduler.step()

    with open('output.txt', 'w') as file:
        original_stdout = sys.stdout
        sys.stdout = file
        for i, item in enumerate(info):
            print(f'Case{i}: average loss: {item[0]:.3f}, average train acc: {item[1]:.3f}, average valid acc: {item[2]:.3f}')
        sys.stdout = original_stdout
    torch.save(best_net, 'ApaFeatureNet.pth')
