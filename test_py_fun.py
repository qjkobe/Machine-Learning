# -*-coding:utf-8-*-
import numpy as np
list1=['neil','mike','lucy']
list2=['123456','xuiasj==','passWD123']
list3=['www.abc.com','www.mike.org','www.lucy.gov']
list0=['name','password','url']
print [dict(zip(list0, l)) for l in zip(list1, list2, list3)]

a = [[3, 4], [4, 5]]
b = [1]
print a[b[0]]
print dict(a)