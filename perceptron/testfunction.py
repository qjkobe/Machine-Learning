
training_set = [([3, 3], 1), ([4, 3], 1), ([1, 1], -1)]
w = [5, 5]
b = 0
# lam is learning rate
lam = 1


def cal(item):
    result = 0
    for i in range(len(item[0])):
        result += item[0][i]*w[i]
    result += b
    result *= item[1]
    # print result
    return result


def update(item):
    # print item
    global w, b
    for i in range(len(item[0])):
        w[i] += lam * item[1] * item[0][i]
    b += lam * item[1]
    print str(w) + str(b)


if __name__ == "__main__":
    while True:
        # when flag is true, there is also misclassification gene
        flag = False
        for item in training_set:
            if cal(item) <= 0:
                update(item)
                flag = True
        if not flag:
            break





