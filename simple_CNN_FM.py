import torch
from IPython import display
import Fashion_MNIST
import numpy as np
import math
from Utility.MyConvolution import ConvolutionLayer
from Utility.MyConvolution import AveragePoolingLayer2X2
from Utility.MyConvolution import DenseLayer

BATCH_SIZE = 1024
NUMBER_CATES = 10
TOTAL_NUMBER = 900
TOTAL_NUMBER_TEST = 150
SAMPLE_RATE = 0.25

alpha = 1.0


def convert(X_unclipped, y_unclipped):
    X_clipped = []
    y_clipped = []
    for i in range(X_unclipped.shape[0]):
        temp = X_unclipped[i, 0].numpy()
        temp = (temp - np.mean(temp)) / np.std(temp)
        X_clipped.append(temp)
        one_hot = np.zeros((NUMBER_CATES, 1))
        one_hot[y_unclipped[i]] += 1
        y_clipped.append(one_hot)
    return X_clipped, y_clipped


def cross_entropy(Y_hat, Y):
    Y_grad = np.zeros((NUMBER_CATES, 1))
    dom = 0
    for i in range(NUMBER_CATES):
        dom += math.exp(Y_hat[i, 0])
    for i in range(NUMBER_CATES):
        Y_grad[i, 0] = math.exp(Y_hat[i, 0]) / dom - Y[i, 0]
    return Y_grad


if __name__ == '__main__':
    train_iter, test_iter = Fashion_MNIST.load_data_fashion_mnist(BATCH_SIZE)

    Conv1 = ConvolutionLayer((28, 28), 1, (5, 5), 6)
    Pooling1 = AveragePoolingLayer2X2((28, 28), 6)
    Conv2 = ConvolutionLayer((14, 14), 6, (5, 5), 6)
    Pooling2 = AveragePoolingLayer2X2((14, 14), 6)
    Dense = DenseLayer((7, 7), 6, 10)

    cnt = 0
    for X_original, y_original in train_iter:
        X, y = convert(X_original, y_original)
        cnt += 1
        for epoch in range(20):
            total_loss = 0
            lr = 0.5
            for i in range(len(X)):
                Conv1.forward([X[i]])
                Pooling1.forward(Conv1.Y)
                Conv2.forward(Pooling1.Y)
                Pooling2.forward(Conv2.Y)
                Dense.forward(Pooling2.Y)

                Y_hat = Dense.Y
                loss = cross_entropy(Y_hat, y[i])
                total_loss += np.linalg.norm(loss)

                Conv1.zero_grad()
                Pooling1.zero_grad()
                Conv2.zero_grad()
                Pooling2.zero_grad()
                Dense.zero_grad()

                Dense.backward(loss)
                Pooling2.backward(Dense.X_grad)
                Conv2.backward(Pooling2.X_grad)
                Pooling1.backward(Conv2.X_grad)
                Conv1.backward(Pooling1.X_grad)

                Conv1.step(lr)
                Conv2.step(lr)
                Dense.step(lr)
            print(f'Average loss: {total_loss / len(X)}', end=' ')
            print(f'learning rate: {lr}')
        alpha = alpha + 0.01
        print(f'{BATCH_SIZE} samples accomplished')
        print('-' * 40)
        break

    cnt = 0
    right, wrong = 0, 0
    for X_original, y_original in train_iter:
        X, y = convert(X_original, y_original)
        cnt += 1
        for i in range(len(X)):
            Conv1.forward([X[i]])
            Pooling1.forward(Conv1.Y)
            Conv2.forward(Pooling1.Y)
            Pooling2.forward(Conv2.Y)
            Dense.forward(Pooling2.Y)

            Y_hat = Dense.Y
            maxi = 0
            for j in range(NUMBER_CATES):
                if Y_hat[j, 0] > Y_hat[maxi, 0]:
                    maxi = j
            if y[i][maxi, 0] == 1:
                right += 1
            else:
                wrong += 1
        break
    print(f'Train acc: {right / (right + wrong)}')

    cnt = 0
    right, wrong = 0, 0
    for X_original, y_original in test_iter:
        X, y = convert(X_original, y_original)
        cnt += 1
        for i in range(len(X)):
            Conv1.forward([X[i]])
            Pooling1.forward(Conv1.Y)
            Conv2.forward(Pooling1.Y)
            Pooling2.forward(Conv2.Y)
            Dense.forward(Pooling2.Y)

            Y_hat = Dense.Y
            maxi = 0
            for j in range(NUMBER_CATES):
                if Y_hat[j, 0] > Y_hat[maxi, 0]:
                    maxi = j
            if y[i][maxi, 0] == 1:
                right += 1
            else:
                wrong += 1
        break
    print(f'Test acc: {right / (right + wrong)}')
