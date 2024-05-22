import numpy
import torch
import torchvision
import matplotlib.pyplot as plt
from Fashion_MNIST import load_data_fashion_mnist
from sklearn import datasets
import numpy as np
import math
from Utility.MyConvolution import ConvolutionLayer
from Utility.MyConvolution import AveragePoolingLayer2X2
from Utility.MyConvolution import DenseLayer

NUMBER_SAMPLES = 200
NUMBER_CATES = 10

lr = 0.08


def convert(points, labels):
    X = []
    Y = []
    for i in range(NUMBER_SAMPLES):
        temp = points[i]
        temp = (temp - np.mean(temp)) / np.std(temp)
        temp = temp.reshape(8, -1)
        X.append(temp)
        temp = np.zeros((NUMBER_CATES, 1))
        temp[labels[i]] = 1
        Y.append(temp)
    return X, Y


def cross_entropy(Y_hat, Y):
    Y_grad = np.zeros((NUMBER_CATES, 1))
    dom = 0
    for i in range(NUMBER_CATES):
        dom += math.exp(Y_hat[i, 0])
    for i in range(NUMBER_CATES):
        Y_grad[i, 0] = math.exp(Y_hat[i, 0]) / dom - Y[i, 0]
    return Y_grad


if __name__ == '__main__':
    digits = datasets.load_digits()
    points = digits.data[:NUMBER_SAMPLES]
    labels = digits.target[:NUMBER_SAMPLES]
    points_test = digits.data[NUMBER_SAMPLES:NUMBER_SAMPLES * 2]
    labels_test = digits.target[NUMBER_SAMPLES:NUMBER_SAMPLES * 2]

    digitfigure = plt.figure(figsize=(12, 2))
    digitfigure.canvas.manager.set_window_title('Dataset')
    for i in range(10):
        subdigiti = digitfigure.add_subplot(1, 10, i + 1)
        subdigiti.imshow(digits.images[i])
        subdigiti.set_title('Digit ' + str(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    X, Y = convert(points, labels)
    X_test, Y_test = convert(points_test, labels_test)

    Conv = ConvolutionLayer((8, 8), 1, (3, 3), 3)
    Pooling = AveragePoolingLayer2X2((8, 8), 3)
    Dense = DenseLayer((4, 4), 3, 10)
    for epoch in range(20):
        total_loss = 0
        for i in range(NUMBER_SAMPLES):
            for j in range(4):
                Conv.forward([X[i]])
                Pooling.forward(Conv.Y)
                Dense.forward(Pooling.Y)

                Y_hat = Dense.Y
                loss = cross_entropy(Y_hat, Y[i])
                if j == 3:
                    total_loss += np.linalg.norm(loss)

                Conv.zero_grad()
                Pooling.zero_grad()
                Dense.zero_grad()

                Dense.backward(loss)
                Pooling.backward(Dense.X_grad)
                Conv.backward(Pooling.X_grad)

                Conv.step(lr)
                Dense.step(lr)
        print(f'Current epoch: {epoch + 1}', end=' ')
        print(f'average loss: {total_loss / NUMBER_SAMPLES}')

    right, wrong = 0, 0
    for i in range(NUMBER_SAMPLES):
        Conv.forward([X[i]])
        Pooling.forward(Conv.Y)
        Dense.forward(Pooling.Y)
        Y_hat = Dense.Y
        maxi = 0
        for j in range(NUMBER_CATES):
            if Y_hat[j, 0] > Y_hat[maxi, 0]:
                maxi = j
        if int(Y[i][maxi, 0]) == 1:
            right += 1
        else:
            wrong += 1
    print(f'Train acc: {right / (right + wrong)}')

    right, wrong = 0, 0
    for i in range(NUMBER_SAMPLES):
        Conv.forward([X_test[i]])
        Pooling.forward(Conv.Y)
        Dense.forward(Pooling.Y)
        Y_hat = Dense.Y
        maxi = 0
        for j in range(NUMBER_CATES):
            if Y_hat[j, 0] > Y_hat[maxi, 0]:
                maxi = j
        if int(Y_test[i][maxi, 0]) == 1:
            right += 1
        else:
            wrong += 1
    print(f'Test acc: {right / (right + wrong)}')
