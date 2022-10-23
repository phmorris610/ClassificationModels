""" Accuracy = TP+TN / n. Precision = TP / TP + FP.
Recall (sensitivity or True Positive Rate) = TP / TP + FN.
F1 Score = 2*precision*recall / precision+recall.
False Positive Rate = FP / PF + TN = 1 - specificity


Numpy resource
In Numpy dimensions are colled axes
[[1., 0., 0.],
 [0., 1., 2.]] has 2 axes, the first has length of 2,
 the second has a length of 3
Numpy array class is called ndarray, not the same as in
Python standard library
ndarray.ndim,  number of axes of the array
ndarray.shape,  the axes of the array, is a tuple of ints
indicating size in each dimension, shape is (n,m), length
is number of axes, ndim
ndarray.size,  total number of elements of array, equal to
the product of the elements of shape
ndarray.dtype,  ex numpy.int32, numpy.int16, numpy.foat64
ndarray.itemsize,  the size of bytes of each element in array
ndarray.data,  the actual elements of the array
"""
import pandas as pd
import numpy as np
import matplotlib as plt
import math

# Numpy examples
# a = np.arange(15).reshape(3, 5)
# print(a)
# print('a\'s shape is', a.shape, ' a\'s dimensions are',
#       a.ndim, ' and a\'s shape is:', a.size)
# # print(a.dtype.name)
# # print(a.itemsize)
# # print(type(a))
# b = np.array([6, 7, 8])
# print(b)
# print('a\'s shape is', b.shape, ' a\'s dimensions are',
#       b.ndim, ' and a\'s shape is:', b.size)
# -----Array Creation----------------------------------
# c = np.array([[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
#              dtype=complex)
# print(c.shape, c.size, c.ndim)
# print(type(c))
# d = np.array([(1, 2, 3, 4), (2, 3, 4, 5)])
# print(type(d))
# e = np.zeros((3, 4))
# f = np.ones((2, 3, 4))
# g = np.empty((2, 3))  # fills acording to memory
# h = np.arange(10, 30, 5)  # starting at 10 no more than 30 by 5s
# i = np.arange(100, 500, 10)  # same
# j = np.linspace(0, 2, 9)  # use for floating numbers
# print(j)
# k = np.random.Generator(20)
# print(k)
# a = np.array([20, 30, 40, 50])
# b = np.arange(4)
# print(a, b)
# c = a - b  # just subtracts down the line
# d = b**2  # just down the line as well
# e = a < 35  # returns a boolean
# print(d, e)
#
# a = np.ones(3, dtype=np.int32)
# b = np.linspace(0, math.pi, 3)
# print(b.dtype.name)  # upcasting, type is now float64
# c = a + b
# d = np.exp(c * 1j)
# print(d)
# print(d.dtype.name)
# compute the sum
# rg = np.random.default_rng(1)
# a = rg.random((2, 3))
# print(a)
# print(a.sum(), a.min(), a.max(), a.mean())
# b = np.arange(12).reshape(2, 6)  # must compatible size and shape
# print(b)
# print(b.cumsum(axis=1))
""" more funtions:
all, any, apply_along_axis, argmax, argmin, argsort, 
average, bincount, ceil, clip, conj, corrcoef, cov, 
cross, cumprod, cumsum, diff, dot, floor, inner, invert, 
lexsort, max, maximum, mean, median, min, minimum, 
nonzero, outer, prod, re, round, sort, std, sum, trace, 
transpose, var, vdot, vectorize, where
"""


# a = np.arange(10)**3
# print(a)
# print(a[2])  # index normally
# print(a[2:5])  # more
# a[:6:2] = 1000
# # from start to position 6,  exclusive, set every 2nd
# # element to a 1000
# print(a[-1])  # for the last element
# for i in a:
#     print(i**(1/3))
# multidimensional arrays


# def f(x, y):
#     return 10 * x + y
#
#
# b = np.fromfunction(f, (5, 4), dtype=int)
# # print(b)
# # print(b[1:3, :])  # each column in the 2nd-3rd row
# # print(b[4, 2])  # remember index starts at 0
# # print(b[-1, -1])  # to get last row and column
# # pin, https://numpy.org/devdocs/user/quickstart.html
# x = np.array([[i * 10]] * 3 for i in range(4))
# print(x)
#
#
# import torch
#
# w = torch.tensor(2.0, requires_grad=True)
# y = 3 * w + 1
# z = y ** 2 + 7
# t = (50 - z) ** 2
# t.backward()
# print(w.grad)
# visual studio
# slice, data science show
# https://github.com/sgeinitz/cs39aa_notebooks
# huggingface.co
# softmax, how to pick the amount of hidden layers
#
#
x = np.array([[1, 2, 4], [2, 3, 4]])
y = np.array([[1, 2, 3]])
z = np.zeros((28, 28, 28))
a = np.zeros((28, 28))
b = np.zeros((28, 1))
c = np.zeros((3, 3))
d = np.zeros((32,))
e = np.zeros((1, 28))
f = np.zeros((28,))
print(z+a)
print(z+b)
print(z+e)
print(z+f)

# copilot for coding



