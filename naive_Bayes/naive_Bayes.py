import numpy as np

X = [[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
     [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
     [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"]]
Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
x = [2, "S"]
X_set = [[1, 2, 3], ["S", "M", "L"]]
Y_set = [1, -1]
PY = np.empty(len(Y_set))
PXY0 = np.empty([len(X_set[0]), len(Y_set)])
PXY1 = np.empty([len(X_set[1]), len(Y_set)])

lam = 1


def training():
    for i in range(len(Y_set)):
        fenmu = len(Y)
        fenzi = 0.
        for j in range(len(Y)):
            if Y[j] == Y_set[i]:
                fenzi += 1

        PY[i] = fenzi / fenmu

        fenmu = fenzi
        for j in range(len(X_set[0])):
            fenzi = 0.
            for k in range(len(X)):
                # print str(X[k][0]) + ":" + str(Y[k])
                if X[k][0] == X_set[0][j] and Y[k] == Y_set[i]:
                    fenzi += 1
            PXY0[j][i] = fenzi / fenmu

        for j in range(len(X_set[1])):
            fenzi = 0.
            for k in range(len(X)):
                # print str(X[k][1]) + ":" + str(X_set[1][j])
                if X[k][1] == X_set[1][j] and Y[k] == Y_set[i]:
                    fenzi += 1
            # print fenzi
            PXY1[j][i] = fenzi / fenmu
    return

    # print PY
    # print PXY0
    # print PXY1


def training_smoothly():
    for i in range(len(Y_set)):
        fenmu = len(Y)
        fenzi = 0.
        for j in range(len(Y)):
            if Y[j] == Y_set[i]:
                fenzi += 1

        PY[i] = (fenzi + lam) / (fenmu + len(Y_set) * lam)

        fenmu = fenzi
        for j in range(len(X_set[0])):
            fenzi = 0.
            for k in range(len(X)):
                # print str(X[k][0]) + ":" + str(Y[k])
                if X[k][0] == X_set[0][j] and Y[k] == Y_set[i]:
                    fenzi += 1
            PXY0[j][i] = (fenzi + lam) / (fenmu + len(X_set[0] * lam))

        for j in range(len(X_set[1])):
            fenzi = 0.
            for k in range(len(X)):
                # print str(X[k][1]) + ":" + str(X_set[1][j])
                if X[k][1] == X_set[1][j] and Y[k] == Y_set[i]:
                    fenzi += 1
            # print fenzi
            PXY1[j][i] = (fenzi + lam) / (fenmu + len(X_set[1] * lam))
    return


if __name__ == "__main__":
    training_smoothly()
    x = [2, "S"]
    x0, x1 = 0, 0
    for i in range(len(X_set[0])):
        if x[0] == X_set[0][i]:
            x0 = i
    for i in range(len(X_set[1])):
        if x[1] == X_set[1][i]:
            x1 = i
    P = np.empty(len(PY))
    max_res = 0
    result = 0
    for i in range(len(PY)):
        P[i] = PY[i] * PXY0[x0][i] * PXY1[x1][i]
        if P[i] > max_res:
            max_res = P[i]
            result = Y_set[i]
    print result
