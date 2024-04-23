import torch
import torch.nn as nn
from Utility.Accumulator import Accumulator
from Utility.Animator import Animator
import Utility.Visualize as UV
import Fashion_MNIST


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


if __name__ == '__main__':
    batch_size = 256
    inputs_size = 28 * 28
    inter_size = 256
    outputs_size = 10
    train_iter, test_iter = Fashion_MNIST.load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(inputs_size, inter_size),
                        nn.ReLU(),
                        nn.Linear(inter_size, outputs_size))

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10

    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        net.eval()
        metric2 = Accumulator(2)
        with torch.no_grad():
            for X, y in test_iter:
                metric2.add(accuracy(net(X), y), y.numel())
        test_acc = metric2[0] / metric2[1]
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        print("Processing: ", float((epoch + 1) / num_epochs) * 100, '%')

    UV.plt.show()
    