# -*-coding:utf-8-*-
# ID3决策树生成算法
from info_gain import *
D = []
A = []
e = 0


class IDNode(object):
    def __init__(self, point, split, mark, subtree):
        self.point = point  # 节点
        self.split = split  # 用来分割的特征
        self.mark = mark  # 类标记
        self.subtree = subtree  # 子树是一个字典，表示特征取不同值的时候，对应的分支


def tree_generate(X, Y, A, e):
    flag = 1
    for i in range(len(Y) - 1):
        if Y[i] != Y[i + 1]:
            flag = 0
            break
    if flag == 1:
        return IDNode(None, None, Y[0], None)
    if not A:
        # 特征为空集，找出个数最大的类。
        count = dict()
        for i in xrange(len(Y)):
            if Y[i] in count:
                count[Y[i]] += 1
            else:
                count[Y[i]] = 1
        return IDNode(None, None, Y[max(count)], None)

    hd = calc_empirical_entropy(X, Y)
    max_info_gain = 0
    for i in xrange(len(A)):
        if calc_info_gain(i, hd, X, Y) > max_info_gain:
            max_info_gain = calc_info_gain(i, hd, X, Y)
            max_index = i
    if max_info_gain < e:
        # 如果信息增益小于阈值，则置T为单节点树，Y中出现最多的类为类标记
        # 特征为空集，找出个数最大的类。
        count = dict()
        for i in xrange(len(Y)):
            if Y[i] in count:
                count[Y[i]] += 1
            else:
                count[Y[i]] = 1
        return IDNode(None, None, Y[max(count)], None)
    # 根据信息增益最大特征的每个可能取值的ai（把这个特征记录下来），将Y分割，保存在di字典里，构建子节点，由子节点及其子节点构成树T
    di = dict()
    for i in xrange(len(X)):
        if X[i][max_index] in di:
            di[X[i][max_index]] += 1
        else:
            di[X[i][max_index]] = 1

    res = 0
    subtree_dic = dict()
    for item in di:
        # a，b分别表示根据取值划分的特征值和相应的类别
        a = []
        b = []
        subtree_dic_x = dict()
        subtree_dic_y = dict()
        for i in xrange(len(X)):
            if item == X[i][max_index]:
                b.append(Y[i])
                a.append(X[i])
        subtree_dic_x[item] = a
        subtree_dic_y[item] = b
        subtree_dic[item] = tree_generate(a, b, A, e)
    return IDNode(None, max_index, None, subtree_dic)


class IDTree(object):
    def __init__(self, X, Y, A, e):
        self.root = tree_generate(X, Y, A, e)


# 为了更好地遍历这棵树，把父节点的分割特征和特征值传下去，并显示深度
def preorder(root, A, split_value, split, depth=0):
    if split != -1 and split_value != -1:
        print "特征值" + str(A[split]) + "?等于" + str(split_value) + "时，" + "深度为" + str(depth)
    if root.split:
        # 如果有分割，先输出分割，且一定有子树
        # print "根据" + str(A[root.split]) + "分割"
        for item in root.subtree:
            preorder(root.subtree[item], A, item, root.split, depth + 1)

    # 如果没有子树，说明是叶节点，输出叶子节点的值
    else:
        print "结果为" + str(root.mark) + " 深度为" + str(depth)


# 使用决策树进行分类
def classify(idtree, test_x):
    if idtree.split:
        return classify(idtree.subtree[test_x[idtree.split]], test_x)
    else:
        # print idtree.mark
        return idtree.mark


if __name__ == "__main__":
    # 根据书上的例子。青年1中年2老年3。是1否0。一般1好2非常好3
    X = [[1, 0, 0, 1], [1, 0, 0, 2], [1, 1, 0, 2], [1, 1, 1, 1], [1, 0, 0, 1],
         [2, 0, 0, 1], [2, 0, 0, 2], [2, 1, 1, 2], [2, 0, 1, 3], [2, 0, 1, 3],
         [3, 0, 1, 3], [3, 0, 1, 2], [3, 1, 0, 2], [3, 1, 0, 3], [3, 0, 0, 1]]
    Y = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    A = ["年龄", "有工作", "有自己的房子", "信贷情况"]
    ID = IDTree(X, Y, A, 0.01)
    preorder(ID.root, A, -1, -1)
    # 测试一下分类效果
    result = []
    for i in xrange(len(X)):
        result.append(classify(ID.root, X[i]))
    print result
    eq_count = 0
    for i in xrange(len(Y)):
        if Y[i] == result[i]:
            eq_count += 1
    print "准确率为：" + str(eq_count / len(Y))
