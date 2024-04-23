import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.datasets import make_moons
from simple_SVM import SimpleSVM
import math
import random


def make_hills(n_samples, noise):
    points, labels = [], []
    x_pos = np.linspace(-1, 2, n_samples)
    for i in range(0, n_samples):
        if random.uniform(0, 1) > 0.5:
            points.append([x_pos[i], math.sin(math.pi / 3 * x_pos[i] + math.pi / 3) - 0.6])
            labels.append(0)
        else:
            points.append([x_pos[i], math.sin(math.pi / 3 * x_pos[i] + math.pi / 3) * 1.5 - 0.3])
            labels.append(1)
    points = np.array(points)
    labels = np.array(labels)
    disturb = np.random.normal(loc=0.0, scale=noise, size=points.shape)
    points = points + disturb
    return points, labels


def mesh_grids(x_lower, x_upper, y_lower, y_upper, nums):
    x_points = np.linspace(x_lower, x_upper, nums)
    y_points = np.linspace(y_lower, y_upper, nums)
    x_grids, y_grids = np.meshgrid(x_points, y_points)
    grids = np.concatenate((np.concatenate(x_grids)[None, :], np.concatenate(y_grids)[None, :])).T
    return x_grids, y_grids, grids


def plot_contourf(main_figure, rows, cols, index, title, style, x_grids, y_grids, y_hat, X_test, y_label):
    sub_figure = main_figure.add_subplot(rows, cols, index)
    sub_figure.set_title(title)
    sub_figure.contourf(x_grids, y_grids, y_hat.reshape(x_grids.shape), cmap=style)
    sub_figure.plot(X_test[:, 0][y_label == 0], X_test[:, 1][y_label == 0], "bo")
    sub_figure.plot(X_test[:, 0][y_label == 1], X_test[:, 1][y_label == 1], "r^")
    sub_figure.grid(True, which='both')


NUM_SAMPLES = 100

if __name__ == '__main__':
    X, y = make_moons(n_samples=NUM_SAMPLES, noise=0.1, random_state=27)
    x1, x2, g = mesh_grids(-1.5, 2.5, -1.0, 1.5, NUM_SAMPLES)

    modelSVM = SVC(kernel='poly', degree=3, coef0=0.2)
    #modelSVM2 = NuSVC(kernel='rbf', gamma='scale', nu=0.1)
    modelSVM.fit(X, y)
    y_predict = modelSVM.predict(g)

    figure = plt.figure(figsize=(8, 8))
    figure.canvas.manager.set_window_title('SVMs')
    plot_contourf(figure, 2, 2, 1, 'Scikit-learn SVM model on moons', 'Blues',
                  x1, x2, y_predict, X, y)

    mySVM = SimpleSVM()
    mySVM.fit(X, y)
    my_y_predict = mySVM.predict(g, X)
    plot_contourf(figure, 2, 2, 3, 'My SVM model on moons', 'Blues',
                  x1, x2, my_y_predict, X, y)

    X, y = make_hills(NUM_SAMPLES, 0.07)
    x1, x2, g = mesh_grids(-1.5, 2.5, -1.0, 1.5, NUM_SAMPLES)

    modelSVM.fit(X, y)
    y_predict = modelSVM.predict(g)
    plot_contourf(figure, 2, 2, 2, 'Scikit-learn SVM model on hills', 'Blues',
                  x1, x2, y_predict, X, y)

    mySVM2 = SimpleSVM()
    mySVM2.fit(X, y)
    my_y_predict = mySVM2.predict(g, X)
    plot_contourf(figure, 2, 2, 4, 'My SVM model on hills', 'Blues',
                  x1, x2, my_y_predict, X, y)

    plt.show()
