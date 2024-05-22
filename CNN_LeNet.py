import torch
from torch import nn

import Utility.Timer
import Utility.Visualize as Uv
from Fashion_MNIST import load_data_fashion_mnist
from Utility.Accumulator import Accumulator
from simple_softmax import accuracy
from Utility.GPU import try_gpu
from Utility.Animator import Animator


net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(400, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)


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


BATCH_SIZE = 256
num_epochs = 10

if __name__ == '__main__':
    test_size = torch.randn(1, 1, 28, 28)
    for layer in net:
        test_size = layer(test_size)
        print(layer.__class__.__name__, 'output shape: \t', test_size.shape)

    train_iter, test_iter = load_data_fashion_mnist(batch_size=BATCH_SIZE)

    net.apply(init_weights)

    device = try_gpu()
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.9)
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
