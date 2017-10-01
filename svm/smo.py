# -*-coding:utf-8-*-
from numpy import *
from sklearn.datasets import load_iris

training_set = array([([3, 3], 1), ([4, 3], 1), ([1, 1], -1)])
x = [[3, 3], [4, 3], [1, 1]]
y = [1, 1, -1]
c = None
alpha = 0


def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros([numSamples, 1]))

    if kernelType == "Linear": # u * v
        kernelValue = matrix_x * sample_x.T
    elif kernelType == "rbf": # exp(|u - v| ^ 2 / 2 * sigma ** 2)
        sigma = kernelOption[1]
        if sigma == 0:
            sigma = 1
        for i in xrange(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        pass
    return kernelValue


def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0] # = len(train_x)
    kernelMatraix = mat(zeros([numSamples, numSamples]))
    for i in xrange(numSamples):
        kernelMatraix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
        return kernelMatraix


class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet # 每行一个样本
        self.train_y = labels  # 相应的标签
        self.C = C             # 松弛变量
        self.toler = toler     # 终止条件
        self.numSamples = dataSet.shape[0] # 样本数量
        self.alphas = mat(zeros((self.numSamples, 1))) # 拉格朗日乘子
        self.b = 0
        self.errorCache = mat(zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)


def calcGg(svm):
    output = array(zeros([svm.numSamples]))

    for i in xrange(svm.numSamples):
        # print i
        output[i] = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, i] + svm.b)
        output[i] *= svm.train_y[i]
    return output - 1


def calcError(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]


def predict(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
    return output_k


# 这里需要修改，修改成计算全部的gg值
def pick1(svm, alpha_i):
    # 得到 alpha > 0 和 alpha < C
    con1 = svm.alphas > 0
    con2 = svm.alphas < svm.C
    con1 = asarray(con1.T)[0]
    con2 = asarray(con2.T)[0]
    # 算出“差异向量”并拷贝成三份
    err1 = calcGg(svm)
    err2 = err1.copy()
    err3 = err1.copy()
    print err1
    # 依次根据三个 KKT 条件，将差异向量的某些位置设为 0
    err1[(con1 & (err1 <= 0)) | (~con1 & (err1 > 0))] = 0
    err2[((~con1 | ~con2) & (err2 != 0)) | ((con1 & con2) & (err2 == 0))] = 0
    err3[(con2 & (err3 >= 0)) | (~con2 & (err3 < 0))] = 0
    # 算出平方和并取出使得“损失”最大的 idx
    err = err1 * err1 + err2 * err2 + err3 * err3
    print err
    idx = argmax(err)
    # 如果小于容忍值，就直接返回
    if err[idx] < svm.toler:
        return
    return idx


# 为什么随机选取。因为这个idx2不随机选取真得是太麻烦了
def pick2(svm, idx1):
    idx = random.randint(svm.numSamples)
    # 这里的 idx1 是第一个参数对应的 idx
    while idx == idx1:
        idx = random.randint(svm.numSamples)
    return idx


def _update_db_cache(svm, idx1, idx2, da1, da2, y1, y2, e1, e2):
    gram_12 = svm.kernelMat[idx1][idx2]
    b1 = -e1 - y1 * svm.kernelMat[idx1][idx1] * da1 - y2 * gram_12 * da2
    b2 = -e2 - y1 * gram_12 * da1 - y2 * svm.kernelMat[idx2][idx2] * da2
    if (0 < svm.alphas[idx1]) and (svm.alphas[idx1] < svm.C) and (0 < svm.alphas[idx2]) and (svm.alphas[idx2] < svm.C) :
        svm.b = b1
    else:
        svm.b = (b1 + b2) / 2


def update_alpha(svm, idx1, idx2):
    if svm.train_y[idx1] != svm.train_y[idx2]:
        L = max(0, svm.alphas[idx2] - svm.alphas[idx1])
        H = min(svm.C, svm.C + svm.alphas[idx2] - svm.alphas[idx1])
    else:
        L = max(0, svm.alphas[idx2] + svm.alphas[idx1] - svm.C)
        H = min(svm.C, svm.alphas[idx2] + svm.alphas[idx1])
    if L == H:
        return 0

    y1, y2 = svm.train_y[idx1], svm.train_y[idx2]
    e1 = predict(svm, idx1) - svm.train_y[idx1]
    e2 = predict(svm, idx2) - svm.train_y[idx2]
    eta = svm.kernelMat[idx1, idx1] + svm.kernelMat[idx2, idx2] - 2 * svm.kernelMat[idx1, idx2]
    a2_new = svm.alphas[idx2] + (y2 * (e1 - e2)) / eta
    if a2_new > H:
        a2_new = H
    elif a2_new < L:
        a2_new = L
    a1_old, a2_old = svm.alphas[idx1], svm.alpha[idx2]
    da2 = a2_old - a2_new
    da1 = da2 * y1 * y2
    # 更新alpha1
    svm.alphas[idx1] += da1
    svm.alphas[idx2] = a2_new


def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('rbf', 1.0)):
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)

    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # 要么到达最大迭代次数，要么就是每个alpha都满足ktt条件。也就是说alpha一圈下来没有变化
    # while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
    #     alphaPairsChanged = 0
    #
    #     if entireSet:
    #         for i in xrange(svm.numSamples):
    while iterCount < maxIter:
        for i in xrange(svm.numSamples):
            idx1 = pick1(svm, i)
            if idx1 is None:
                break
            idx2 = pick2(svm, idx1)
            update_alpha(svm, idx1, idx2)

    return svm


def testSVM(svm, test_x, test_y):
    svm = set_string_function()
    test_x = mat(x)
    test_y = mat(y)
    numTestSamples = test_x.shape[0]
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    supportVectors = svm.train_x[supportVectorsIndex]
    supportVectorLabels = svm.train_y[supportVectorsIndex]
    supportVectorAlphas = svm.alphas[supportVectorsIndex]
    matchCount = 0
    for i in xrange(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
        predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        if sign(predict) == sign(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    print accuracy


if __name__ == "__main__":
    train_x = mat(x)
    train_y = mat(y).T
    print train_x
    print train_y
    test_x = mat(x)
    test_y = mat(y).T

    C = 0.6
    toler = 0.01
    maxIter = 50
    svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))

    print testSVM(svmClassifier, test_x, test_y)


def get_lower_bound(idx1, idx2):
    if y[idx1] != y[idx2]:
        return max(0., alpha[idx2] - alpha[idx1])
    return max(0., alpha[idx2] + alpha[idx1] - c)


def get_upper_bound(self, idx1, idx2):
    if self._y[idx1] != self._y[idx2]:
        return min(self._c, self._c + self._alpha[idx2] - self._alpha[idx1])
    return min(self._c, self._alpha[idx2] + self._alpha[idx1])
