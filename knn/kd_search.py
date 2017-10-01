from math import sqrt
from collections import namedtuple
import numpy as np

result = namedtuple("result", ["nearest_point", "nearest_dist", "nodes_visited"])

# test = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
test = np.array([[1, 2, 3], [2, 3, 4]])
uu = test.tolist()
gg = np.std(uu)
# print gg
# for i in range(len(uu)):
#     print uu[i][0] ** 2 + uu[i][1] ** 2


def find_nearest(tree, point):
    k = len(point)

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)

        nodes_visited = 1
        s = kd_node.split
        p = kd_node.point

        if target[s] <= p[s]:
            nearer_node = kd_node.left
            futher_node = kd_node.right
        else:
            nearer_node = kd_node.right
            futher_node = kd_node.left

        temp = travel(nearer_node, target, max_dist)
        # print temp
        # print max_dist
        nearest = temp.nearest_point
        dist = temp.nearest_dist

        nodes_visited += temp.nodes_visited

        if dist < max_dist:
            max_dist = dist
# if distance is nearer, renew the distance.
        temp_dist = abs(p[s] - target[s])
        if max_dist < temp_dist:
            return result(nearest, dist, nodes_visited)

        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(p, target)))

        if temp_dist < dist:
            nearest = p
            dist = temp_dist
            max_dist = dist

        temp2 = travel(futher_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))

