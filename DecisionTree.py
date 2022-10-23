import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# --------load the Regression Model first----------------------------------------------------------------------
data_URL = 'https://raw.githubusercontent.com/sgeinitz/cs39aa_notebooks/main/data/trainB1.csv'
data = pd.read_csv(data_URL)
train_data = data
null_entries = data.isnull().sum()
# print(null_entries)
cols_w_nulls = null_entries[null_entries > 0].index
X = train_data.drop(columns=cols_w_nulls).copy()
Y = train_data.pop('SalePrice')
# encode the categoricals
for colname in X.select_dtypes('object').columns:
    X[colname], _ = X[colname].factorize()

featured_selections = ['2ndFlrSF', 'BsmtFinSF1', 'PoolArea', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'GrLivArea']
X_linreg = X[featured_selections].copy()  # using the selections glean from the Chi-Squared test
linreg_model = LinearRegression()
linreg_model.fit(X_linreg, Y)
y_pred = linreg_model.predict(X_linreg)
#
dectree_model = DecisionTreeRegressor(criterion='squared_error')
# dectree_model = RandomForestRegressor(criterion='mse', n_estimators=500)
dectree_model.fit(X_linreg, Y)
y_pred = dectree_model.predict(X_linreg)
# Plot the predictions side-by-side with the actual values
# Note: a perfect model would yield the identity, y=x
plt.scatter(Y, y_pred, alpha=0.3)
plt.xlabel('y actual')
plt.ylabel('y predicted')
plt.plot([0, 5e5], [0, 5e5], c='k', alpha=0.2)
# plt.show()
# Calculate and show the mean squared error
mse_linreg = (((Y - y_pred) ** 2).mean())
print("mse_linreg = ", mse_linreg)
print("rmse_linreg = ", np.sqrt(mse_linreg))
# To avoid overfitting we need to tinker with the hyperparameter max_depth, how will it work for out-of-sample data
# For this we will split into train vs test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_linreg, Y, test_size=0.2, random_state=1)
# Fit lin regression and calculate predicted test y's
linreg_model.fit(X_train, y_train)
y_linreg_train_pred = linreg_model.predict(X_train)
y_linreg_pred = linreg_model.predict(X_test)
# Fit lin regression and calculate predicted test y's
dectree_model = DecisionTreeRegressor(criterion='squared_error',
                                      random_state=1, min_samples_leaf=12, max_depth=10)#, min_samples_leaf=12)
dectree_model.fit(X_train, y_train)
y_dectree_train_pred = dectree_model.predict(X_train)
y_dectree_pred = dectree_model.predict(X_test)
# Plot the predictions side-by-side with the actual values
# Note: a perfect model would yield the identity, y=x
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
# ax1.scatter(y_train, y_linreg_train_pred, alpha=0.2, c='b')
# ax1.scatter(y_train, y_dectree_train_pred, alpha=0.2, c='r')
# ax1.set(title="Training Dataset (in-sample)", xlabel='y train actual', ylabel='y train predicted')
# ax1.legend(["Linear Regression", "Decision Tree"], loc=4)
# ax1.plot([0, 5e5], [0, 5e5], c='k', alpha=0.2)
# plt.show()

# ax2.scatter(y_test, y_linreg_pred, alpha=0.2, c='b')
# ax2.scatter(y_test, y_dectree_pred, alpha=0.2, c='r')
# ax2.set(title="Test Dataset (out-of-sample)", xlabel='y test actual', ylabel='y test predicted')
# ax2.legend(["Linear Regression", "Decision Tree"], loc=4)
# ax2.plot([0, 5e5], [0, 5e5], c='k', alpha=0.2)
# plt.show()

# Calculate and show the mean squared errors for the test dataset
print("For the test dataset:")
mse_linreg = (((y_test - y_linreg_pred) ** 2).mean())
print(f"    lin reg mse = {mse_linreg:12.2f} (rmse = {np.sqrt(mse_linreg):.2f})")

mse_dectree = (((y_test - y_dectree_pred) ** 2).mean())
print(f"    dectree mse = {mse_dectree:12.2f} (rmse = {np.sqrt(mse_dectree):.2f})")

print(f"    decision tree loss {100 * (mse_dectree / mse_linreg):.2f}% of linear regression loss")
#
""" If the model does well out-of-sample it "Generalizes Well"
Now we can begin to tweak the hyperparameters  max_depth or min_samples_leaf, 
which states the minimum number of training observations that will be used (and averaged) 
for any leaf in the decision tree"""
#
# plotting the tree with sklearn
# still haven't found an easy way....
