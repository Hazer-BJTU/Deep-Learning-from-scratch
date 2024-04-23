import torch
import torchvision
import matplotlib.pyplot as plt
from Fashion_MNIST import load_data_fashion_mnist
from sklearn import datasets
import numpy as np
import math

NUMBER_SAMPLES = 500
NUMBER_CATES = 10


class SimplePCA:
    def __init__(self, dimension=3, K=5):
        self.dimension = dimension
        self.W = None
        self.K = K

    def fit(self, X, Y):
        mat = X.T @ X
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        sortlst = []
        for i in zip(eigenvalues, eigenvectors):
            sortlst.append(i)
        sortlst.sort(key=lambda t: -t[0])
        W = []
        for i in range(0, self.dimension):
            W.append(sortlst[i][1].tolist())
        self.W = np.array(W).T

    def map(self, X):
        Z = self.W.T @ X.T
        return Z

    def reset(self):
        self.W = None

    def predict(self, samples, inputs, targets):
        y_hat = []
        for i in range(inputs.shape[0]):
            votes = [0 for _ in range(NUMBER_CATES)]
            lst = []
            for j in range(samples.shape[0]):
                dist = np.linalg.norm(samples[j] - inputs[i])
                if len(lst) < self.K:
                    lst.append(j)
                else:
                    for k in range(len(lst)):
                        if np.linalg.norm(samples[lst[k]] - inputs[i]) > dist:
                            lst[k] = j
                            break
            for k in range(len(lst)):
                votes[targets[lst[k]]] += 1
            index = 0
            for k in range(NUMBER_CATES):
                if votes[k] > votes[index]:
                    index = k
            y_hat.append(index)
        return np.array(y_hat)


if __name__ == '__main__':
    '''
    batch_size = 1
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    cnt = NUMBER_SAMPLES
    points, labels = [], []
    for x, y in train_iter:
        x_array = x.flatten().numpy()
        x_array = (x_array - np.mean(x_array)) / np.std(x_array)
        points.append(x_array.tolist())
        labels.append(y.item())
        cnt -= 1
        if cnt == 0:
            break
    points = np.array(points)
    '''

    digits = datasets.load_digits()
    points = digits.data[:NUMBER_SAMPLES]
    labels = digits.target[:NUMBER_SAMPLES]
    points_test = digits.data[NUMBER_SAMPLES:NUMBER_SAMPLES * 2]
    labels_test = digits.target[NUMBER_SAMPLES:NUMBER_SAMPLES * 2]

    for i in range(NUMBER_SAMPLES):
        x = points[i]
        points[i] = (x - np.mean(x)) / np.std(x)
        x = points_test[i]
        points_test[i] = (x - np.mean(x)) / np.std(x)

    #pcaModel = SimplePCA(2, 1)
    pcaModel = SimplePCA(16, 3)
    pcaModel.fit(points, labels)
    points2D = pcaModel.map(points)
    points2D_test = pcaModel.map(points_test)

    figure = plt.figure(figsize=(8, 8))
    digitfigure = plt.figure(figsize=(12, 2))
    figure.canvas.manager.set_window_title('PCA mapping')
    digitfigure.canvas.manager.set_window_title('Dataset')
    subfigure = figure.add_subplot(1, 1, 1)
    subfigure.set_title('Digits in 2D')

    for i in range(10):
        subdigiti = digitfigure.add_subplot(1, 10, i + 1)
        subdigiti.imshow(digits.images[i])
        subdigiti.set_title('Digit ' + str(i))
        plt.xticks([])
        plt.yticks([])

    points2D = np.array(list(map(lambda p: p.real, points2D.T)))
    points2D_test = np.array(list(map(lambda p: p.real, points2D_test.T)))
    for i in range(NUMBER_CATES):
        xs, ys = [], []
        for j in range(NUMBER_SAMPLES):
            if labels[j] == i:
                xs.append(points2D[j, 0])
                ys.append(points2D[j, 1])
        subfigure.scatter(xs, ys)
    subfigure.grid(True, which='both')
    plt.show()

    y_hat = pcaModel.predict(points2D, points2D_test, labels)
    y_hat_train = pcaModel.predict(points2D, points2D, labels)
    predict_true, predict_false = 0, 0
    predict_true_train, predict_false_train = 0, 0
    for i in range(NUMBER_SAMPLES):
        if y_hat[i] == labels_test[i]:
            predict_true += 1
        else:
            predict_false += 1
        if y_hat_train[i] == labels[i]:
            predict_true_train += 1
        else:
            predict_false_train += 1
    acc = predict_true / (predict_true + predict_false)
    acc_train = predict_true_train / (predict_true_train + predict_false_train)
    print(f'Test accuracy: {acc}')
    print(f'Train accuracy: {acc_train}')
