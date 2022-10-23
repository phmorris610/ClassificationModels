import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression

data = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/Housing.csv')

# print(data.columns)
# print(type(data))  # printing the type of the data
# print("train_df.shape: ", data.shape)  # printing the shape of the data
# print(
#     f"train_df has {data.shape[0]} rows and {data.shape[1]} columns")  # x amount of rows and x amount of columns
# print(data.describe())  # printing a summary of the statistics
# print(data.columns)
import matplotlib.pyplot as plt

# plt.hist(data['SalePrice'], bins=20, edgecolor='black')
# plt.xlabel("home sale price")
# plt.ylabel("# of observations")
# plt.show()
# plt.scatter(data['1stFlrSF'], data['SalePrice'], alpha=0.3)
# plt.xlabel("first floor square footage")
# plt.ylabel("home sale price")
# plt.show()
train_data = data
null_entries = data.isnull().sum()
# print(null_entries)
cols_w_nulls = null_entries[null_entries > 0].index
X = train_data.drop(columns=cols_w_nulls).copy()
Y = train_data.pop('SalePrice')
# encode the categoricals
for colname in X.select_dtypes('object').columns:
    X[colname], _ = X[colname].factorize()
# Next feature selection
# -------------First the chi-squared stat test---------------
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# extracting top 10 best features by applying SelectKBest class
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # this names the df columns
print(featureScores.nlargest(10, 'Score'))  # prints 10 best features
# -------------Now the Extra Tree Classifier----------------------
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
# -------------Now we will do correlation stats with a Heatmap------
import seaborn as sns
print(data.columns)
corrmat = data.corr()
top_corr_feat = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(data[top_corr_feat].corr(), annot=True, cmap='RdYlGn')
plt.show()
# -------------Now the regression-----------------------------------
featured_selections = ['2ndFlrSF', 'BsmtFinSF1', 'PoolArea', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'GrLivArea']
X_linreg = X[featured_selections].copy()  # using the selections glean from the Chi-Squared test
linreg_model = LinearRegression()
linreg_model.fit(X_linreg, Y)
y_pred = linreg_model.predict(X_linreg)
import statsmodels.api as sm
regModel = sm.OLS(Y, X_linreg).fit()
print(regModel.summary())  # summary statistics for the regression model
# From the summary we can see by the p-Values all these variable are significant,
# although the 2ndFlrSF variable is borderline. Another measure, R^2 being .94, tells
# us that these variable are highly correlated, also that F-stat is very large
# Now lets take a peek at the Residuals
plt.scatter(y_pred, Y-y_pred)
plt.show()
# We can see that the data is normal, although there looks to be an increasing variance
# -------------------Now asses more-----------------------------------
y_pred = linreg_model.predict(X_linreg)  # to see the predicted y values
# Plot the predictions side-by-side with the actual values
# Note: a perfect model would yield the identity, y=x
plt.scatter(Y, y_pred, alpha=0.3)
plt.xlabel('y actual')
plt.ylabel('y predicted')
plt.plot([0, 5e5], [0, 5e5], c='k', alpha=0.2)
# plt.show()
# Manually calculate and show the mean squared error
mse_linreg = (((Y - y_pred)**2).mean())
print("mse_linreg = ", mse_linreg)
print("rmse_linreg = ", np.sqrt(mse_linreg))
# --------------------------we have manipulated, plotted, plotted and assessed feature selection, calculated
# summary statistics then plotted results, y predicted vs actual y values



