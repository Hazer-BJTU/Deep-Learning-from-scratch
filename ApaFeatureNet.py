import torch
import math
import argparse
import sys
import copy

from torch import nn
from CIFAR_TEN import load_data_cifar_10
from Utility.GPU import try_gpu
from AdaptivePolynomialApproximator import AdaptivePolynomialApproximator


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class ApaFeatureNet(nn.Module):
    def __init__(self, **kwargs):
        super(ApaFeatureNet, self).__init__(**kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.Flatten()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.Flatten()
        )

        self.linear1 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh()
        )
        self.APA1 = AdaptivePolynomialApproximator(256, 3)
        self.APA2 = AdaptivePolynomialApproximator(256, 3)
        self.APA3 = AdaptivePolynomialApproximator(256, 3)
        self.drop_out = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, X):
        features1 = self.conv1(X)
        features2 = self.conv2(X)
        features = torch.cat((features1, features2), dim=1)

        apainput = self.linear1(features)
        linearout = self.linear2(features)
        apaout1 = self.drop_out(self.APA1(apainput) + linearout)
        apaout2 = self.drop_out(self.APA2(apainput) + linearout)
        apaout3 = self.drop_out(self.APA3(apainput) + linearout)
        output = torch.cat((apaout1, apaout2, apaout3), dim=1)

        return self.classifier(output)


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
    parser.add_argument('--num_epochs', type=int, nargs='?', default=120)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.85)
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
