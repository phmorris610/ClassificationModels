# Using the Titanic Dataset
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

# data_URL = 'https://raw.githubusercontent.com/sgeinitz/cs39aa_notebooks/main/data/trainB2.csv'
train_df = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/titanic3.csv')
# train_df = pd.read_csv(data_URL)
print("train_df.shape: ", train_df.shape)
print(train_df.columns)
# print(train_df.describe())
null_entries = train_df.isnull().sum()
# print(null_entries)
# remove cabin columns
train_df.drop(columns=["Cabin"], inplace=True)
# remove rows with missing Age/Embarked values
train_df.dropna(axis=0, inplace=True)
train_df.reset_index(inplace=True)  # an important line, resets the index after dropping the null values
# check shape of data after removing those rows/cols
# print(train_df.shape)
X = train_df.copy()
y = X.pop("Survived")
# Encoding categoricals
for colname in X.select_dtypes('object').columns:
    X[colname], _ = X[colname].factorize()
# Feature selection again
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(X, y)
# print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
# plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.scatter(X.index, y, alpha=0.2)
ax1.set_title('Survival ~ index')
ax2.scatter(X.PassengerId, y, alpha=0.2)
ax2.set_title('Survival ~ PassengerId')
# plt.show()
# ------------------A quick regression to see the p-values of these variables--------------------------------
# featured_selections = ['SibSp', 'Fare', 'Pclass', 'Age', 'Sex']
# X_linreg = X[featured_selections].copy()
# linreg_model = LinearRegression()
# linreg_model.fit(X_linreg, y)
# y_pred = linreg_model.predict(X_linreg)
# import statsmodels.api as sm
#
# regModel = sm.OLS(y, X_linreg).fit()
# print(regModel.summary())  # summary statistics for the regression model
""" parch and name (especially name) and ticket are not good for the model, but index and passengerID have
very low p-values so the above plots against the y showing in incosequence is important"""
# removing PassengerId and index greatly improved the model
# y_pred = linreg_model.predict(X_linreg)
# mse_linreg = (((y - y_pred)**2).mean())
# print("mse_linreg = ", mse_linreg)
# print("rmse_linreg = ", np.sqrt(mse_linreg))
# show the MSE and rMSE and move on to RandomForest
# A collection of predictive models is called an ensemble
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X, y)
y_pred = rf_model.predict_proba(X)
# Plot the predictions side-by-side with the actual values
# Note: a perfect model would yield the identity, y=x
plt.scatter(y, y_pred[:, 1], alpha=0.3)
plt.xlabel('y actual')
plt.ylabel('y predicted (probability)')
plt.plot([0, 1], [0.5, 0.5], c='k', alpha=0.2)
plt.show()
# We will use MSE as a loss function, but also Cross Entropy
# Formula for CE: CE(y,y_hat) = (1/n)*log(y_hat)
yhat = y_pred[:, 1]
# Calculate mean-squared error and binary cross entropy
rf_mse = (((y - y_pred[:, 1]) ** 2).mean())
print(f"mse: {rf_mse:.4f}")
# Calculate BCE manually
p0 = np.sum((1.00 - y[y == 0].to_numpy()) * np.log(1.00 - yhat[y == 0]))
p1 = np.sum(y[y == 1].to_numpy() * np.log(yhat[y == 1]))
rf_bce = -1 * (p1 + p0)
print(f"manual calc of bce: {rf_bce:.4f}")
# Calculate BSE (aka log-loss) using scikit learn
from sklearn.metrics import log_loss

print(f"bce w/ sklearn: {log_loss(y, yhat, normalize=False):.4f}")
# Now lets split the data
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, random_state=1)


def fitRFModel(min_samples_split_hyper_param):
    rf_model = RandomForestClassifier(min_samples_split=min_samples_split_hyper_param, random_state=1)
    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict_proba(X_train)[:, 1]
    y_val_pred = rf_model.predict_proba(X_val)[:, 1]
    train_loss = log_loss(y_train,
                          y_train_pred)  # , F.binary_cross_entropy(torch.tensor(y_train_pred), torch.tensor(y_train.to_numpy().astype(float)), reduction="mean")
    val_loss = log_loss(y_val,
                        y_val_pred)  # F.binary_cross_entropy(torch.tensor(y_val_pred), torch.tensor(y_val.to_numpy().astype(float)), reduction="mean")
    # return((train_loss.item(), val_loss.item()))
    return (train_loss, val_loss)


# Possible values of min_samples_split are 10 to 70 (by 5)
hyp_param_vals = list(range(5, 151, 5))
# hyp_param_vals = [5] + list(range(10,201,10))
losses = []
for hp in hyp_param_vals:
    losses.append(fitRFModel(hp))
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0, 0, 1, 1])  # .1, 0.1, 0.8, 0.8]) # main axes
ax.plot(hyp_param_vals, [loss[1] for loss in losses], '--ro')  # validattion loss
ax.plot(hyp_param_vals, [loss[0] for loss in losses], '--bo')  # training loss
ax.legend(["Validation Loss", "Train Loss"], loc=4)
ax.set_xticks(hyp_param_vals)
ax.set(xlabel="min samples to split", ylabel="loss (lower is better)")
plt.show()
# mse: 0.0206
# manual calc of bce: 83.2701
# binary cross entropy w/ sklearn: 83.2701
# As we can see the best hyperparameter is 39