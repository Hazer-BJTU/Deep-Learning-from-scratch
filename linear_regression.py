import random
import torch
import matplotlib.pyplot as plt


def synthetic_data(_w, _b, num_examples):
    x = torch.normal(0, 1, (num_examples, len(_w)))
    y = torch.matmul(x, _w) + _b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape(-1, 1)


def data_iter(_batch_size, _features, _labels):
    num_examples = len(_features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, _batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + _batch_size, num_examples)]
        )
        yield _features[batch_indices], _labels[batch_indices]


def linreg(_x, _w, _b):
    return torch.matmul(_x, _w) + b


def squared_loss(y_hat, _y):
    return (y_hat - _y) ** 2 / 2


def sgd(params, _lr, _batch_size):
    with torch.no_grad():
        for param in params:
            param -= _lr * param.grad / _batch_size
            param.grad.zero_()


def paint_line(_w, _b, c):
    num_w = _w.detach().numpy()
    num_b = _b.detach().numpy()
    plt.plot([-3, 3], [-3 * num_w[1, 0] + num_b[0], 3 * num_w[1, 0] + num_b[0]], color = c)


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)

    batch_size = 10
    for X, Y in data_iter(batch_size, features, labels):
        print(X, '\n', Y)
        break

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    for epoch in range(num_epochs):
        for X, Y in data_iter(batch_size, features, labels):
            L = loss(net(X, w, b), Y)
            L.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_L = loss(net(features, w, b), labels)
            print('epoch ', epoch, ', loss ', train_L.mean())

    print('true_w - w: ', true_w - w.reshape(true_w.shape))
    print('true_b - b: ', true_b - b)
    paint_line(w, b, 'r')
    true_w = true_w.reshape(-1, 1)
    true_b = torch.tensor([true_b])
    paint_line(true_w, true_b, 'g')
    plt.show()
