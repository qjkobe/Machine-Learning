import numpy as np
from math import sqrt
from sklearn.datasets import load_iris


data_set = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
labels = ['A', 'A', 'B', 'B']
k = 3


def classify(input, data_set, label, k):
    data_size = len(data_set)
    diff = np.tile(input, (data_size, 1)) - data_set

    dist = np.sum(diff ** 2, axis=1) ** 0.5
    sorted_dist_index = np.argsort(dist)

    for i in range(data_size):
        dist[i] = sqrt(np.sum((x1 - x2) ** 2 for x1, x2 in zip(input, data_set[i])))

    class_count = {}
    for i in range(k):
        vote_label = label[sorted_dist_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    max_count = 0
    classes = label[0]
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            classes = key

    return classes


if __name__ == "__main__":
    # point = np.array([1.1, 0.3])
    # point = np.array([0.1, 3])
    # ans = classify(point, data_set, labels, k)
    # print ans
    iris = load_iris()
    print len(iris.data)
    data_set = np.array(iris.data[10:])
    labels = np.array(iris.target[10:])

    for i in range(10):
        point = np.array(iris.data[i])
        ans = classify(point, data_set, labels, k)
        print "expected: " + str(iris.target_names[iris.target[i]]) +\
              " actually: " + str(iris.target_names[ans])
