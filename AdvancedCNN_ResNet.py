import torch
from torch import nn
from torch.nn import functional as func

import Utility.Timer
import Utility.Visualize as Uv
from Fashion_MNIST import load_data_fashion_mnist
from Utility.Accumulator import Accumulator
from simple_softmax import accuracy
from Utility.GPU import try_gpu
from Utility.Animator import Animator
from CIFAR_TEN import load_data_cifar_10


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = func.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            Y += self.conv3(X)
        else:
            Y += X
        return func.relu(Y)


def res_net(first_input_channels, args):
    layers = [nn.Conv2d(first_input_channels, 64, kernel_size=7, stride=2, padding=3),
              nn.BatchNorm2d(64), nn.ReLU(),
              nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

    for (input_channels, num_channels, num_residuals, isfirst) in args:
        for i in range(num_residuals):
            if i == 0:
                if not isfirst:
                    layers.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
                else:
                    layers.append(Residual(input_channels, num_channels))
            else:
                layers.append(Residual(num_channels, num_channels))

    return nn.Sequential(*layers,
                         nn.AdaptiveAvgPool2d((1, 1)),
                         nn.Flatten(),
                         nn.Linear(512, 10))


net = res_net(3, ((64, 64, 2, True), (64, 128, 2, False),
                  (128, 256, 2, False), (256, 512, 2, False)))


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if device is None:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


BATCH_SIZE = 128
num_epochs = 10
learning_rate = 0.05

if __name__ == '__main__':
    test_size = torch.randn(1, 3, 96, 96)
    for layer in net:
        test_size = layer(test_size)
        print(layer.__class__.__name__, 'output shape: \t', test_size.shape)

    # train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE, resize=224)
    train_iter, test_iter = load_data_cifar_10(batch_size=BATCH_SIZE, resize=96)

    net.apply(init_weights)

    device = try_gpu()
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer = Utility.Timer.Timer()
    num_batches = len(train_iter)
    metric = Accumulator(3)

    for epoch in range(num_epochs):
        net.train()
        for i, (X_orig, y_orig) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X_orig.to(device), y_orig.to(device)
            y_hat = net(X)
            L = loss(y_hat, y)
            L.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(L * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'Epoch {epoch + 1} accomplished with test accuracy {test_acc:.3f}')

    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

    Uv.plt.show()
