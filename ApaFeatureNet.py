import torch
import math
import argparse
import sys
import copy

from torch import nn
from CIFAR_100 import load_data_cifar_100
from Utility.GPU import try_gpu


def batch_quadratic(X1, A, X2, batch_size, input_features, output_features):
    assert X1.shape == X2.shape and X1.shape[1] == input_features
    assert A.shape[0] == input_features * output_features and A.shape[1] == input_features
    Y = X1 @ A.T
    Y = Y.view(batch_size, output_features, input_features)
    output = torch.bmm(Y, torch.unsqueeze(X2, dim=2))
    return torch.squeeze(output, dim=2)


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
            ResBlock1(256, 256, (5, 5)),
            ResBlock1(256, 256, (5, 5)),
            ResBlock2(256, 256, (5, 5)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock2(256, 256, (3, 3)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock1(256, 256, (3, 3)),
            ResBlock2(256, 256, (3, 3)),
            nn.Dropout(0.25)
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
        self.W1 = nn.Sequential(
            nn.Linear(input_size, hiddens),
            nn.Tanh()
        )
        self.W2 = nn.Linear(input_size, output_size)
        self.W3 = nn.Linear(rank * hiddens, output_size)
        self.output_layer = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.25)
        )
        self.params = nn.Parameter(torch.randn(rank, hiddens * hiddens, hiddens))
        self.BNZ = nn.BatchNorm1d(hiddens)

    def forward(self, X):
        Z = self.W1(X)
        Zi = Z
        output = None
        for i in range(self.rank):
            Zi = batch_quadratic(Zi, self.params[i], Z, X.shape[0], self.hiddens, self.hiddens)
            Zi = self.BNZ(Zi)
            if output is None:
                output = Zi
            else:
                output = torch.cat((output, Zi), dim=1)
        return self.output_layer(self.W2(X) + self.W3(output))


class ApaFeatureNet(nn.Module):
    def __init__(self, pretrainPath, device, **kwargs):
        super(ApaFeatureNet, self).__init__(**kwargs)
        self.featureExtraction = torch.load(pretrainPath, map_location=device, weights_only=False)
        self.flatten = nn.Flatten()
        self.APA = ApaBlock(1024, 128, 1024, 5)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1536),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(1536, 1536),
            nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(1536, 100)
        )

    def forward(self, X):
        F = self.flatten(self.featureExtraction(X))
        return self.classifier(self.APA(F))


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


def train(net, lr, num_epochs, weight_decay, train_iter, valid_iter, device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
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
    parser.add_argument('--num_epochs', type=int, nargs='?', default=300)
    parser.add_argument('--lr', type=float, nargs='?', default=0.001)
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0001)
    parser.add_argument('--batch_size', type=int, nargs='?', default=256)
    parser.add_argument('--phase', type=int, nargs='?', default=0)
    parser.add_argument('--model_path', type=str, nargs='?', default='FeatureExtraction.pth')
    parser.add_argument('--model_path2', type=str, nargs='?', default='ApaFeatureNet.pth')
    args = parser.parse_args()

    train_iter, valid_iter = load_data_cifar_100(args.batch_size, None)
    if args.phase == 0:
        net = PreTrainModel()
        net.apply(init_weights)
        best_net, info = train(net, args.lr, args.num_epochs, args.weight_decay,
                               train_iter, valid_iter, try_gpu(args.cuda_idx))
        torch.save(net.Feature, 'FeatureExtraction.pth')
        write_info(info, 'Pretrain_output.txt')
    elif args.phase == 1:
        net2 = ApaFeatureNet(args.model_path, try_gpu(args.cuda_idx))
        best_net2, info2 = train(net2, args.lr / 3, args.num_epochs // 3, args.weight_decay,
                                 train_iter, valid_iter, try_gpu(args.cuda_idx))
        torch.save(net2, 'ApaFeatureNet.pth')
        write_info(info2, 'train_output.txt')
    else:
        net3 = torch.load(args.model_path2, map_location=try_gpu(args.cuda_idx))
        best_net3, info3 = train(net3, args.lr / 50, args.num_epochs // 3, args.weight_decay,
                                 train_iter, valid_iter, try_gpu(args.cuda_idx))
        torch.save(net3, 'ApaFeatureNet.pth')
        write_info(info3, 'train_output.txt')
