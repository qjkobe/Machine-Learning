# -*-coding:utf-8-*-
# 信息增益算法
# 样本X，对应类Y
import math
import numpy as np
X = []
Y = []


def calc_empirical_entropy(X, Y):
    # 计算经验熵。先计算每一类（Y取值）的概率，各个取值结果保存在ck{}里，最终结果是返回值
    ck = {}
    len_x = len(X)
    len_y = len(Y)
    # X和Y的长度其实是相等的。这里假设数据集相同下标XY取值对应
    for i in xrange(len_y):
        if Y[i] in ck:
            ck[Y[i]] += 1
        else:
            ck[Y[i]] = 1
    res = 0
    for item in ck:
        res += -1.0 * (1.0 * ck[item] / len_x) * math.log((1.0 * ck[item] / len_x), 2)
    return res


def calc_empirical_condi_entropy(num):
    # 计算第i个特征值对应的每个取值di，再计算这每个取值下对应Y中各个类别的数量dik，并求出hdi。最后求出这个特征值对应的经验条件熵
    len_x = len(X)
    len_y = len(Y)
    di = {}

    for i in xrange(len_x):
        if X[i][num] in di:
            di[X[i][num]] += 1
        else:
            di[X[i][num]] = 1

    res = 0
    for item in di:
        # a，b分别表示根据取值划分的特征值和相应的类别
        a = []
        b = []
        for i in xrange(len_x):
            if item == X[i][num]:
                b.append(Y[i])
                a.append(X[i])
        res += 1.0 * (1.0 * len(b) / len_x) * calc_empirical_entropy(a, b)

    return res


def calc_info_gain(num, hd):
    # 结果是第i个特征值对应的信息增益
    res = 0
    res += hd
    res -= calc_empirical_condi_entropy(num)
    return res


if __name__ == "__main__":
    # 根据书上的例子。青年1中年2老年3。是1否0。一般1好2非常好3
    X = [[1, 0, 0, 1], [1, 0, 0, 2], [1, 1, 0, 2], [1, 1, 1, 1], [1, 0, 0, 1],
         [2, 0, 0, 1], [2, 0, 0, 2], [2, 1, 1, 2], [2, 0, 1, 3], [2, 0, 1, 3],
         [3, 0, 1, 3], [3, 0, 1, 2], [3, 1, 0, 2], [3, 1, 0, 3], [3, 0, 0, 1]]
    Y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    hd = calc_empirical_entropy(X, Y)
    max_info_gain = 0
    for i in range(len(X[0])):
        print calc_info_gain(i, hd)
        if calc_info_gain(i, hd) > max_info_gain:
            max_info_gain =  calc_info_gain(i, hd)
            max_index = i
    print "最终结果是：" + str(max_info_gain) + "应该选择第" + str(max_index) + "个特征"
