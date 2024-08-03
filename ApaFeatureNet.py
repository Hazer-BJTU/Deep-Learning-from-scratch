import torch
import math
import argparse
import sys
import copy

from torch import nn
from CIFAR_100 import load_data_cifar_100
from Utility.GPU import try_gpu
from AdaptivePolynomialApproximator import AdaptivePolynomialApproximator
from AdaptivePolynomialApproximator import batch_quadratic


class ResBlock1(nn.Module):
    def __init__(self, input_channels, output_channels, kernels_size, **kwargs):
        super(ResBlock1, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.resblock = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernels_size[0], stride=1, padding='same'),
            nn.BatchNorm2d(output_channels), nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernels_size[1], stride=1, padding='same'),
            nn.BatchNorm2d(output_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, X):
        F = self.resblock(X)
        return self.relu(F + X)


class ResBlock2(nn.Module):
    def __init__(self, input_channels, output_channels, kernels_size, **kwargs):
        super(ResBlock2, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.resblock = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernels_size[0], stride=1, padding='same'),
            nn.BatchNorm2d(output_channels), nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernels_size[1], stride=2, padding=kernels_size[1] // 2),
            nn.BatchNorm2d(output_channels)
        )
        self.conv = nn.Conv2d(input_channels, output_channels, 1, stride=2)
        self.relu = nn.ReLU()

    def forward(self, X):
        F = self.resblock(X)
        return self.relu(F + self.conv(X))


class FeatureExtraction(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtraction, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, 9, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.resblocks = nn.Sequential(
            ResBlock1(256, 256, (3, 3)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock2(256, 256, (3, 3)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock2(256, 256, (3, 3)),
            nn.AvgPool2d(kernel_size=2, stride=2), nn.Dropout(0.25)
        )

    def forward(self, X):
        return self.resblocks(self.conv(X))


class PreTrainModel(nn.Module):
    def __init__(self, **kwargs):
        super(PreTrainModel, self).__init__(**kwargs)
        self.Feature = FeatureExtraction()
        self.Classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 1536),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(1536, 1536),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(1536, 100)
        )

    def forward(self, X):
        return self.Classifier(self.Feature(X))


class ApaBlock(nn.Module):
    def __init__(self, input_size, hiddens, output_size, rank, **kwargs):
        super(ApaBlock, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.rank = rank
        self.linear1 = nn.Sequential(
            nn.Linear(input_size, hiddens),
            nn.ReLU(), nn.Dropout(0.25)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(), nn.Dropout(0.25)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(hiddens, output_size),
            nn.ReLU(), nn.Dropout(0.25)
        )
        self.params = nn.Parameter(torch.randn(rank, hiddens * hiddens, hiddens))
        self.BNZ = nn.BatchNorm1d(hiddens)
        self.BNY = nn.BatchNorm1d(hiddens)
        self.outlayer = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.25)
        )

    def forward(self, X):
        Z = self.linear1(X)
        Zi = Z
        Y = torch.zeros(X.shape[0], self.hiddens, device=X.device)
        for i in range(self.rank):
            Zi = batch_quadratic(Zi, self.params[i], Z, X.shape[0], self.hiddens, self.hiddens)
            Zi = self.BNZ(Zi)
            Y = Y + (Zi / self.rank)
        Y = self.BNY(Y)
        return self.outlayer(self.linear3(Y) + self.linear2(X))


class ApaFeatureNet(nn.Module):
    def __init__(self, pretrainPath, device, **kwargs):
        super(ApaFeatureNet, self).__init__(**kwargs)
        self.featureExtraction = torch.load(pretrainPath, map_location=device)
        self.flatten = nn.Flatten()
        self.APA1 = ApaBlock(1024, 256, 512, 3)
        self.APA2 = ApaBlock(1024, 256, 512, 3)
        self.APA3 = ApaBlock(1024, 256, 512, 3)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(2048, 2048),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(2048, 100)
        )

    def forward(self, X):
        F = self.flatten(self.featureExtraction(X))
        apaout1 = self.APA1(F)
        apaout2 = self.APA2(F)
        apaout3 = self.APA3(F)
        apaout = torch.cat((apaout1, apaout2, apaout3), dim=1)
        return self.classifier(apaout)


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


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


def train(net, lr, num_epochs, weight_decay, train_iter, valid_iter, device, pretrain=False):
    net.to(device)
    net.apply(init_weights)
    if pretrain:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam([
            {'params': net.featureExtraction.parameters(), 'lr': lr / 10},
            {'params': net.APA1.parameters(), 'lr': lr},
            {'params': net.APA2.parameters(), 'lr': lr},
            {'params': net.APA3.parameters(), 'lr': lr},
            {'params': net.classifier.parameters(), 'lr': lr}
        ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(num_epochs // 6, 1), 0.6)
    loss = nn.CrossEntropyLoss()
    best_net, best_valid_acc, info = None, 0, []
    print(f'training on {device}')
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc, valid_acc, num_samples = 0, 0, 0, 0
        for X_orig, y_orig in train_iter:
            X, y = X_orig.to(device), y_orig.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            L = loss(y_hat, y)
            L.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            train_loss += L
            num_samples += X.shape[0]
            with torch.no_grad():
                train_acc += accuracy(y_hat, y)
        valid_acc = evaluate(net, valid_iter, device)
        scheduler.step()
        print(f'epoce: {epoch}, average loss: {train_loss / num_samples:.3f}, '
              f'average train acc: {train_acc / num_samples:.3f}, '
              f'average valid acc: {valid_acc:.3f}')
        if valid_acc > best_valid_acc and epoch + 1 >= num_epochs // 2:
            best_valid_acc = valid_acc
            best_net = copy.deepcopy(net)
            info.append((train_loss / num_samples, train_acc / num_samples, valid_acc))
    return best_net, info


def write_info(info, fileName):
    with open(fileName, 'w') as file:
        orig_stdout = sys.stdout
        sys.stdout = file
        for i, item in enumerate(info):
            print(f'Case{i}: average loss: {item[0]:.3f}, average train acc: {item[1]:.3f}, '
                  f'average valid acc: {item[2]:.3f}')
        sys.stdout = orig_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose device')
    parser.add_argument('--cuda_idx', type=int, nargs='?', default=0)
    parser.add_argument('--num_epochs', type=int, nargs='?', default=240)
    parser.add_argument('--lr', type=float, nargs='?', default=0.001)
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0001)
    parser.add_argument('--batch_size', type=int, nargs='?', default=256)
    parser.add_argument('--phase', type=int, nargs='?', default=0)
    parser.add_argument('--model_path', type=str, nargs='?', default='FeatureExtraction.pth')
    args = parser.parse_args()

    train_iter, valid_iter = load_data_cifar_100(args.batch_size, None)
    if args.phase == 0:
        net = PreTrainModel()
        best_net, info = train(net, args.lr, args.num_epochs, args.weight_decay,
                               train_iter, valid_iter, try_gpu(args.cuda_idx), True)
        torch.save(net.Feature, 'FeatureExtraction.pth')
        write_info(info, 'Pretrain_output.txt')
    else:
        net2 = ApaFeatureNet(args.model_path, try_gpu(args.cuda_idx))
        best_net2, info2 = train(net2, args.lr, args.num_epochs, args.weight_decay,
                                 train_iter, valid_iter, try_gpu(args.cuda_idx))
        torch.save(net2, 'ApaFeatureNet.pth')
        write_info(info2, 'train_output.txt')
