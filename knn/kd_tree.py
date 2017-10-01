from kd_search import *
# import kd_search
import numpy as np
from sklearn.datasets import load_iris
import random


# data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
iris = load_iris()
data = np.array(iris.data).tolist()

# print np.std(data, axis=0)

# create some interesting data
data = np.empty([100, 2])
for i in range(len(data)):
    for j in range(len(data[i])):
        if j:
            data[i][j] = random.uniform(0, 10)
        else:
            data[i][j] = random.uniform(0, 1)
data = data.tolist()
print data
data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]).tolist()
k = len(data[0])


class KdNode(object):
    def __init__(self, point, split, left, right):
        self.point = point # the point
        self.split = split # the dimension to partition
        self.left = left
        self.right = right


def cal_deviation(data_set):
    std = np.std(data_set, axis=0)
    max_std = 0
    max_split = 0
    for i in range(len(std)):
        if std[i] > max_std:
            max_std = std[i]
            max_split = i
    return max_split


def create_node2(split, data_set):
    if not len(data_set):
        return None
    # print split
    # print data_set

    data_set.sort(key=lambda x: x[split])
    split_pos = len(data_set) // 2
    median = data_set[split_pos]
    split_next = cal_deviation(data_set)
    # split_next = (split + 1) % k

    return KdNode(median, split,
                  create_node2(split_next, data_set[: split_pos]),
                  create_node2(split_next, data_set[split_pos + 1:]))


def create_node(split, data_set):
    if not len(data_set):
        return None
    # print split
    # print data_set

    data_set.sort(key=lambda x: x[split])
    split_pos = len(data_set) // 2
    median = data_set[split_pos]
    split_next = (split + 1) % k

    return KdNode(median, split,
                  create_node(split_next, data_set[: split_pos]),
                  create_node(split_next, data_set[split_pos + 1:]))


class KdTree(object):
    def __init__(self, data, flag):
        if flag:
            self.root = create_node2(cal_deviation(data), data)
        else:
            self.root = create_node(0, data)


def preorder(root):
    print root.point
    if root.left:
        preorder(root.left)
    if root.right:
        preorder(root.right)


if __name__ == "__main__":
    kd = KdTree(data, 0)
    # preorder(kd.root)
    ans = find_nearest(kd, [1, 3])
    print ans

    kd = KdTree(data, 1)
    ans = find_nearest(kd, [1, 3])
    print ans

