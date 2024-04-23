import numpy as np
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.datasets import make_moons
import math


def kernel(degree, x1, x2):
    product = x1 @ x2[:, None]
    return (1 + product[0]) ** degree


class SimpleSVM:
    def __init__(self, degree=3, epoch_num=10):
        self.degree = degree
        self.alphas = None
        self.b = None
        self.y = None
        self.epoch_num = epoch_num

    def fit(self, points, labels):
        n = points.shape[0]
        self.alphas = np.random.normal(loc=0.0, scale=1.0, size=n)
        self.b = np.random.rand()
        self.y = [1 if i == 1 else -1 for i in labels]

        for epoch in range(self.epoch_num):
            print(f'Starting epoch:{epoch}...')
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    Fp, Fq = 0.0, 0.0
                    for i in range(n):
                        Fp += self.alphas[i] * self.y[i] * kernel(self.degree, points[i], points[p])
                        Fq += self.alphas[i] * self.y[i] * kernel(self.degree, points[i], points[q])
                    Ep, Eq = self.y[p] - Fp, self.y[q] - Fq
                    kpp = kernel(self.degree, points[p], points[p])
                    kqq = kernel(self.degree, points[q], points[q])
                    kpq = kernel(self.degree, points[p], points[q])
                    ap_new = self.alphas[p] + self.y[p] * (Ep - Eq) / (kpp + kqq - 2 * kpq)
                    C = self.alphas[p] * self.y[p] + self.alphas[q] * self.y[q]
                    aq_new = (C - self.y[p] * ap_new) * self.y[q]
                    if ap_new < 0:
                        ap_new = 0
                        aq_new = C * self.y[q]
                    elif aq_new < 0:
                        aq_new = 0
                        ap_new = C * self.y[p]
                    Fp_new = Fp + self.y[p] * (kpp - kpq) * (ap_new - self.alphas[p])
                    Fq_new = Fq + self.y[p] * (kpq - kqq) * (ap_new - self.alphas[p])
                    bp_new = self.y[p] - Fp_new
                    bq_new = self.y[q] - Fq_new
                    self.alphas[p] = ap_new
                    self.alphas[q] = aq_new
                    self.b = (bp_new + bq_new) / 2

    def predict(self, points, train_points):
        answer = []
        n = points.shape[0]
        m = self.alphas.size
        for i in range(n):
            F = self.b
            for j in range(m):
                F += self.alphas[j] * self.y[j] * kernel(self.degree, train_points[j], points[i])
            if F >= 0:
                answer.append(1)
            else:
                answer.append(0)
        return np.array(answer)


NUM_SAMPLES = 100

if __name__ == '__main__':
    X, y = make_moons(n_samples=NUM_SAMPLES, noise=0.1, random_state=27)
    mySVM = SimpleSVM()
    mySVM.fit(X, y)
    y_hat = mySVM.predict(X, X)
    print(y)
    print(y_hat)
