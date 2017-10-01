# -*-coding:utf-8-*-
from numpy import *

mat1 = mat(zeros([5, 5]))
ee = mat([6, 7])
print nonzero(ee.T >= 0)
print ee - 1
aa = mat([[2, 3], [3, 4]])
print float(multiply(aa, aa))
# print aa
# print aa.A
# print aa[0]
print nonzero(aa.A > 0)
print aa > 0
print aa.A > 0
# print asarray(aa)[0]
aa = array([2, 2, 3])
print aa > 0
# print aa.A
print len(ee)
ee += 3
print mat1
mat1 += 3
print mat1 * mat1
# for i in xrange(5):
#     mat1[:, i] =
for i in xrange(5):
    mat1[:, i] = 3
print exp(mat1)