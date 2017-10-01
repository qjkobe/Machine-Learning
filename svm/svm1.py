# -*-coding:utf-8-*-
# 线性支持向量机
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
# x = [[3, 3], [4, 3], [1, 1]]
# y = [1, 1, -1]
# lam is learning rate
lam = 0.01
c = 1
a = np.array(iris.target[:100])
a[:50] -= 1

x, y = np.asarray(iris.data[:100], np.float32), np.asarray(a, np.float32)
w = np.zeros(x.shape[1])
b = 0


def predict(x, raw=False):
    x = np.asarray(x, np.float32)
    y_pred = x.dot(w) + b
    if raw:
        return y_pred
    return np.sign(y_pred).astype(np.float32)


if __name__ == "__main__":

    for _ in range(10000):
        err = 1 - y * predict(x, True)
        idx = np.argmax(err)
        # for idx in range(3):
        if err[idx] <= 0:
            continue
        delta = lam * c * y[idx]
        w += delta * x[idx]
        b += delta

    print str(w) + ":" + str(b)

    test = [6.1,3.0,4.6,1.4]
    print w.dot(test) + b
