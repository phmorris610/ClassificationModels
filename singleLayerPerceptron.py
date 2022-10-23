# # Logistic Regression
# import random
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# """We'll first import the necessary Python modules
# and then generate synthetic data with appropriate
# size/dimensions.
#
# Let's now plot the generated data to see x and y.
# Note that the y outcomes/labels belong to one of two classes.
#  The positive cases are shown in blue while negative cases
#  are in red"""
#
# N = 100  # total number of observations
# D_in = 1  # input dimension (i.e. dimension of a single observation's x vector)
# D_out = 1  # output dimension (i.e. y), so just 1 for this example
# random.seed(1)
# np.random.RandomState(1)
#
# # Create random input data and derive the 'true' labels/output
# x = np.random.randn(N, D_in) + 1
#
#
# def true_y(x_in, n_obs):
#     def addNoise(x):
#         if abs(x - 1) < 1:
#             return 0.1
#         elif abs(x - 1) < 0.1:
#             return 0.25
#         else:
#             return 0.01
#
#     return np.apply_along_axis(lambda x: [int(x < 1) if random.random() < addNoise(x) else int(x > 1)], 1, x_in)
#
#
# y = true_y(x, N).flatten()
#
# plt.scatter(x[y == 1, 0], y[y == 1], c='blue', s=100, alpha=0.2)
# plt.scatter(x[y == 0, 0], y[y == 0], c='red', s=100, alpha=0.2)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(('positive cases', 'negative cases'))
# # plt.show()
#
# """Let's quickly peek at the x and y data objects
# (i.e. numpy arrays) to see what their size,
# shape, and rank look like
# """
# print(f"y.size: {y.size}")
# print(f"y.shape: {y.shape}")
# print(f"y.ndim: {y.ndim}")
#
# print(f"x.size: {x.size}")
# print(f"x.shape: {x.shape}")
# print(f"x.ndim: {x.ndim}")
#
# # fit the logistic regression
# # transfrom the target, Y (a binary variable) by
# # looking at the log odds (logit) of E(y),
# # if we let E(y) be the probability that yi = 1 then
# # the log-odds = ln(mue/1-mue_i)
# # to make a prediction for a new y_i we'll then use
# # the inverse of the logit function, which is the sigmoid
# # 1/(1+e^(b0+b1(x)))
#
# from sklearn.linear_model import LogisticRegression
#
# logreg_model = LogisticRegression(random_state=42, max_iter=100, tol=1e-3, solver='liblinear')
# logreg_model.fit(x, y)
#
# print(f" beta0 = {logreg_model.intercept_[0]:.4f}")
# print(f" beta1 = {logreg_model.coef_[0][0]:.4f}")
# y_pred = logreg_model.predict_proba(x)
# lr_loss = 1 / N * np.square(y - y_pred[:, 1]).sum()
# print(f" loss (mse) = {lr_loss:.4f}")
#
# # b1s = np.arange(6, -4.1, -0.5)
# # b0s = np.arange(-6, 4.1, 0.5)
# b1s = np.arange(9, -5, -0.5)
# b0s = np.arange(-9, 5, 0.5)
# surf = np.array(
#     [[1 / N * np.square(y - 1 / (1 + np.exp(-1 * (b1s[i] * x[:, 0] + b0s[j])))).sum() for j in range(len(b0s))] for i in
#      range(len(b1s))])
# df = pd.DataFrame(surf, columns=b0s, index=b1s)
# p1 = sns.heatmap(df, cbar_kws={'label': 'loss'}, cmap="RdYlGn_r")
# plt.xlabel("beta0")
# plt.ylabel("beta1")
# # plt.show()
# y = np.array([-1,-10])
# x = np.array([[1,2,3],[4,5,6]])
# z = np.array([[-1,-10]])
#
# c = x+z
# v = x*z
# print(v.shape)

y     = 1+1+1
y_hat = 0.3+0.2+0.9+1
mse = (1/4)*(y - y_hat)
print(mse)

TP = 90
FP = 70
TN = 80
FN = 10

precision = TP/(TP+FP)
recall    = TP/(TP+FN)
F1        = (2*precision*recall)/(precision+recall)
print(precision, recall)
n1 = 90+70
n2 = 10+80
print(n2)
print(n1)

import cv
