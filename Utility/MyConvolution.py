import numpy as np
import math


def calc_conv(input_mat, kernel, padding=None):
    if padding is None:
        padding = ((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2)

    row_padding = int(padding[0])
    col_padding = int(padding[1])
    rows = int(row_padding * 2 + input_mat.shape[0])
    cols = int(col_padding * 2 + input_mat.shape[1])
    output_rows = int(rows - kernel.shape[0] + 1)
    output_cols = int(cols - kernel.shape[1] + 1)

    output_mat = np.zeros((output_rows, output_cols), dtype=float)
    for i in range(output_rows):
        for j in range(output_cols):
            for x in range(i, i + kernel.shape[0]):
                for y in range(j, j + kernel.shape[1]):
                    idx = x - row_padding
                    idy = y - col_padding
                    if idx >= input_mat.shape[0] or idx < 0 or idy >= input_mat.shape[1] or idy < 0:
                        continue
                    output_mat[i, j] += input_mat[idx, idy] * kernel[x - i, y - j]
    return output_mat


def calc_grad(input_mat, kernel, output_grad, padding=None):
    if padding is None:
        padding = ((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2)

    row_padding = int(padding[0])
    col_padding = int(padding[1])
    rows = int(row_padding * 2 + input_mat.shape[0])
    cols = int(col_padding * 2 + input_mat.shape[1])
    output_rows = int(rows - kernel.shape[0] + 1)
    output_cols = int(cols - kernel.shape[1] + 1)

    input_grad = np.zeros(input_mat.shape)
    kernel_grad = np.zeros(kernel.shape)
    for i in range(output_rows):
        for j in range(output_cols):
            for x in range(i, i + kernel.shape[0]):
                for y in range(j, j + kernel.shape[1]):
                    idx = x - row_padding
                    idy = y - col_padding
                    if idx >= input_mat.shape[0] or idx < 0 or idy >= input_mat.shape[1] or idy < 0:
                        continue
                    input_grad[idx, idy] += output_grad[i, j] * kernel[x - i, y - j]
                    kernel_grad[x - i, y - j] += output_grad[i, j] * input_mat[idx, idy]
    return input_grad, kernel_grad


class ConvolutionLayer:
    def __init__(self, input_size, input_channels, kernel_size, output_channels):
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        output_prototype = calc_conv(np.zeros(input_size), np.zeros(kernel_size))
        output_size = (output_prototype.shape[0], output_prototype.shape[1])

        self.kernels = []
        self.kernels_grads = []
        self.bias = []
        self.bias_grads = []
        for i in range(output_channels):
            kernel_groups = []
            kernel_groups_grads = []
            for j in range(input_channels):
                temp_kernel = np.random.randn(kernel_size[0], kernel_size[1])
                temp_kernel = (temp_kernel - np.mean(temp_kernel)) / np.std(temp_kernel)
                temp_kernel /= temp_kernel.shape[0] * temp_kernel.shape[1]
                kernel_groups.append(temp_kernel)
                kernel_groups_grads.append(np.zeros(kernel_size))
            self.kernels.append(kernel_groups)
            self.kernels_grads.append(kernel_groups_grads)
            self.bias.append(np.zeros(output_size))
            self.bias_grads.append(np.zeros(output_size))

        self.X = []
        self.X_grad = []
        self.Y = []
        self.Y_grad = []
        for i in range(input_channels):
            self.X.append(np.zeros(input_size))
            self.X_grad.append(np.zeros(input_size))
        for i in range(output_channels):
            self.Y.append(np.zeros(output_size))
            self.Y_grad.append(np.zeros(output_size))

    def forward(self, Y_former):
        for i in range(self.input_channels):
            for j in range(self.X[i].shape[0]):
                for k in range(self.X[i].shape[1]):
                    self.X[i][j, k] = Y_former[i][j, k]
        for i in range(self.output_channels):
            self.Y[i] = np.zeros(self.Y[i].shape)
            for j in range(self.input_channels):
                self.Y[i] += calc_conv(self.X[j], self.kernels[i][j])
            self.Y[i] += self.bias[i]

    def zero_grad(self):
        for i in range(self.input_channels):
            self.X_grad[i] = np.zeros(self.X[i].shape)
        for i in range(self.output_channels):
            for j in range(self.input_channels):
                self.kernels_grads[i][j] = np.zeros(self.kernel_size)
            self.bias_grads[i] = np.zeros(self.Y[i].shape)

    def backward(self, X_grad_successor):
        for i in range(self.output_channels):
            for j in range(self.Y_grad[i].shape[0]):
                for k in range(self.Y_grad[i].shape[1]):
                    self.Y_grad[i][j, k] = X_grad_successor[i][j, k]
        for i in range(self.output_channels):
            for j in range(self.input_channels):
                input_grad_delta, kernel_grad_delta = calc_grad(self.X[j], self.kernels[i][j], self.Y_grad[i])
                self.X_grad[j] += input_grad_delta
                self.kernels_grads[i][j] += kernel_grad_delta
            self.bias_grads[i] += self.Y_grad[i]

    def step(self, lr):
        for i in range(self.output_channels):
            for j in range(self.input_channels):
                self.kernels[i][j] -= lr * self.kernels_grads[i][j]
            self.bias[i] -= lr * self.bias_grads[i]


class AveragePoolingLayer2X2:
    def __init__(self, input_size, input_channels):
        self.input_channels = input_channels
        self.output_size = (int(input_size[0]) // 2, int(input_size[1]) // 2)

        self.X = []
        self.X_grad = []
        for i in range(input_channels):
            self.X.append(np.zeros(input_size))
            self.X_grad.append(np.zeros(input_size))

        self.Y = []
        self.Y_grad = []
        for i in range(input_channels):
            self.Y.append(np.zeros(self.output_size))
            self.Y_grad.append(np.zeros(self.output_size))

    def forward(self, Y_former):
        for i in range(self.input_channels):
            for j in range(self.X[i].shape[0]):
                for k in range(self.X[i].shape[1]):
                    self.X[i][j, k] = Y_former[i][j, k]
        for i in range(self.input_channels):
            self.Y[i] = np.zeros(self.Y[i].shape)
            for j in range(self.Y[i].shape[0]):
                for k in range(self.Y[i].shape[1]):
                    self.Y[i][j, k] += (self.X[i][2 * j, 2 * k] + self.X[i][2 * j + 1, 2 * k]
                                        + self.X[i][2 * j, 2 * k + 1] + self.X[i][2 * j + 1, 2 * k + 1]) * 0.25

    def zero_grad(self):
        for i in range(self.input_channels):
            self.X_grad[i] = np.zeros(self.X[i].shape)

    def backward(self, X_grad_successor):
        for i in range(self.input_channels):
            for j in range(self.Y_grad[i].shape[0]):
                for k in range(self.Y_grad[i].shape[1]):
                    self.Y_grad[i][j, k] = X_grad_successor[i][j, k]
        for i in range(self.input_channels):
            self.Y[i] = np.zeros(self.Y[i].shape)
            for j in range(self.Y[i].shape[0]):
                for k in range(self.Y[i].shape[1]):
                    self.X_grad[i][2 * j, 2 * k] += 0.25 * self.Y[i][j, k]
                    self.X_grad[i][2 * j + 1, 2 * k] += 0.25 * self.Y[i][j, k]
                    self.X_grad[i][2 * j, 2 * k + 1] += 0.25 * self.Y[i][j, k]
                    self.X_grad[i][2 * j + 1, 2 * k + 1] += 0.25 * self.Y[i][j, k]


class DenseLayer:
    def __init__(self, input_size, input_channels, output_size):
        self.input_channels = input_channels
        self.output_size = output_size
        self.vector_size = input_channels * input_size[0] * input_size[1]
        self.X = np.zeros((self.vector_size, 1))
        self.X_grad = []
        for i in range(input_channels):
            self.X_grad.append(np.zeros(input_size))
        self.Y = np.zeros((output_size, 1))
        self.Y_grad = np.zeros((output_size, 1))

        self.W = np.random.randn(output_size, self.vector_size)
        self.W = (self.W - np.mean(self.W)) / np.std(self.W)
        self.W_grad = np.zeros((output_size, self.vector_size))

        self.bias = np.zeros((output_size, 1))
        self.bias_grad = np.zeros((output_size, 1))

    def forward(self, Y_former):
        for i in range(self.input_channels):
            for j in range(Y_former[i].shape[0]):
                for k in range(Y_former[i].shape[1]):
                    idx = i * Y_former[i].shape[0] * Y_former[i].shape[1] + j * Y_former[i].shape[1] + k
                    self.X[idx, 0] = Y_former[i][j, k]
        self.Y = self.W @ self.X + self.bias

    def zero_grad(self):
        for i in range(self.input_channels):
            self.X_grad[i] = np.zeros(self.X_grad[i].shape)
        self.bias_grad = np.zeros((self.output_size, 1))
        self.W_grad = np.zeros((self.output_size, self.vector_size))

    def backward(self, X_grad_successor):
        for i in range(self.output_size):
            self.Y_grad[i, 0] = X_grad_successor[i, 0]
        temp_X_grad = self.W.T @ self.Y_grad
        for i in range(self.input_channels):
            for j in range(self.X_grad[i].shape[0]):
                for k in range(self.X_grad[i].shape[1]):
                    idx = i * self.X_grad[i].shape[0] * self.X_grad[i].shape[1] + j * self.X_grad[i].shape[1] + k
                    self.X_grad[i][j, k] = temp_X_grad[idx, 0]
        self.W_grad += self.Y_grad @ self.X.T
        self.bias_grad += self.Y_grad

    def step(self, lr):
        self.W -= lr * self.W_grad
        self.bias -= lr * self.bias_grad


if __name__ == '__main__':
    C = ConvolutionLayer((28, 28), 1, (3, 3), 3)
    P1 = AveragePoolingLayer2X2((28, 28), 3)
    P2 = AveragePoolingLayer2X2((14, 14), 3)
    D = DenseLayer((7, 7), 3, 10)

    X = np.random.randn(28, 28)
    Y = np.zeros((10, 1))
    Y[5, 0] = 1
    print(X)
    print(Y)

    for epoch in range(50):
        C.forward([X])
        P1.forward(C.Y)
        P2.forward(P1.Y)
        D.forward(P2.Y)

        Y_hat = D.Y
        diff = Y_hat - Y
        print(np.linalg.norm(diff))

        C.zero_grad()
        P1.zero_grad()
        P2.zero_grad()
        D.zero_grad()

        D.backward(diff)
        P2.backward(D.X_grad)
        P1.backward(P2.X_grad)
        C.backward(P1.X_grad)

        lr = 0.05
        D.step(lr)
        C.step(lr)

    print(D.Y)
