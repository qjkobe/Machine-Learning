import numpy as np

training_set = np.array([([3, 3], 1), ([4, 3], 1), ([1, 1], -1)])
a = [0, 0, 0]
b = 0
Gram = np.empty((3,3))
y = np.array(training_set[:, 1])
x = np.empty((3, 2))
lam = 1
for i in range(len(x)):
    x[i] = training_set[i][0]


def cal_gram(x):
    global Gram
    for i in range(len(x)):
        for j in range(len(x)):
            Gram[i][j] = np.dot(x[i], x[j])
    print Gram


def cal(i):
    result = 0
    result += np.dot(a * y, Gram[i])
    result += b
    result *= y[i]
    return result


def update(i):
    global a, b
    a[i] = a[i] + lam
    b = b + lam * y[i]
    print str(a) + str(b)
    return

if __name__ == "__main__":
    cal_gram(x)
    while True:
        # There is also misclassification gene
        flag = False
        for i in range(len(x)):
            if cal(i) <= 0:
                update(i)
                flag = True
        if not flag:
            w = np.dot(a * y, x)
            b = np.dot(a, y)
            print w
            print b
            break
