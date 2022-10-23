import pandas as pd
import numpy as np


class regModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def reg(self):
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        linreg_model = LinearRegression()
        linreg_model.fit(self.X, self.y)
        import statsmodels.api as sm
        regModel = sm.OLS(self.y, self.X).fit()
        y_pred = linreg_model.predict(self.X)
        # and a residual plot
        plt.title('Residulas')
        plt.scatter(regModel.model.exog[:, 1], regModel.resid)
        plt.show()
        mse_linreg = ((y_pred ** 2).mean())
        print("MSE = ", mse_linreg)
        print("rMSE = ", np.sqrt(mse_linreg))
        return regModel.summary()  # finally return a Regression Summary

    def chiSquareTest(self, k):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        bestfeatures = SelectKBest(score_func=chi2, k=k)
        fit = bestfeatures.fit(self.X, self.y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(self.X.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']
        return featureScores.nlargest(k, 'Score')

    def extraTree(self, n):
        from sklearn.ensemble import ExtraTreesClassifier
        import matplotlib.pyplot as plt
        model = ExtraTreesClassifier()
        model.fit(self.X, self.y)
        feat_importances = pd.Series(model.feature_importances_, index=self.X.columns)
        feat_importances.nlargest(n).plot(kind='barh')
        plt.show()
        feat = []
        feat.append(feat_importances.nlargest(n))
        return feat

    def heatmap(self):
        from sklearn.linear_model import LinearRegression
        import seaborn as sns
        import matplotlib.pyplot as plt
        corrmat = data.corr()
        top_corr_feat = corrmat.index
        plt.figure(figsize=(20, 20))
        g = sns.heatmap(data[top_corr_feat].corr(), annot=True, cmap='RdYlGn')
        plt.show()

    def trainTestSplit(self, t_size, state):
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, train_size=t_size, random_state=state)
        return X_train, X_val, y_train, y_val

    def normal(self, feature):
        import matplotlib.pyplot as plt
        plt.hist(self.X[feature], bins=20, edgecolor='black')
        plt.show()  # TODO: I would like to somehow spit out a display of all the hists at once of all of my features

    def tranform(self, selections, trans):
        import numpy as np
        X_tlog = self.X[selections].applymap(lambda x: np.log(x+1))
        y_tlog = np.log(y)
        return X_tlog, y_tlog

    def describe(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X, self.y, alpha=0.3)  #TODO: I would like to display all X's vs the y's
        plt.show()

# Loading the data, seperate explanatory X and response y
data = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/Housing.csv')
train_data = data
null_entries = data.isnull().sum()
# print(null_entries)
cols_w_nulls = null_entries[null_entries > 0].index
X = train_data.drop(columns=cols_w_nulls).copy()
y = train_data.pop('SalePrice')
for colname in X.select_dtypes('object').columns:
    X[colname], _ = X[colname].factorize()
row, col = X.shape
featured_selections = ['GrLivArea', 'BsmtFinSF1', 'LotArea', 'GarageArea', '1stFlrSF',
                       'YearRemodAdd', 'BsmtFinSF1', 'Neighborhood', 'WoodDeckSF',
                       'OverallQual', 'Exterior1st']
X_linreg = X[featured_selections].copy()

# None are but if we have categoricals use this to encode them


# feature extraction
features = regModel(X, y)
xTree = regModel(X, y)
# heat = regModel(X, y)  # not used because too gaudy here
# Split the data
# Split = regModel(X, y)  # to split
# X_train, X_val, y_train, y_val = Split.trainTestSplit(0.80, 1)  # An 80/20 split with rand state = 1
# XTrain = X_train[featured_selections].copy()  # TODO: why can't I pop XTrain into the model; wrong shape?

# print(XTrain.shape)
# print(y_train.shape)
# print(type(XTrain))  # Shape and type are the same
# print(type(y_train))

# Let's check if the data is normal, that Residual plot looks odd.
trans_selections = ['GrLivArea', 'BsmtFinSF1', 'LotArea', 'GarageArea', '1stFlrSF', 'WoodDeckSF',
                    'OpenPorchSF', 'TotalBsmtSF', 'TotRmsAbvGrd', 'Exterior2nd',
                     'Exterior1st', 'YearRemodAdd', 'Neighborhood', 'OverallQual']
X_trans = X[trans_selections].copy()
X_log = X_trans[trans_selections].applymap(lambda x: np.log(x+1))
y_log = np.log(y)
#
norm = regModel(X_log, y_log)
# print(norm.normal('Exterior1st'))  # to look at a some features, it looks like out skewed data (unless it is a zero 1 or
# 2 to indicate a finished basement or a front porch are normal now



# Used for Feature Extraction
# print(features.chiSquareTest(30))
# print(xTree.extraTree(30))
#
# Obviously sale price is correlated with itself but this gives us a good baseline regarding a top score
# maybe we want to go with the variables less than 0.2, now append the featured_selections (also don't include Id,
# or year sold)

# features2 = regModel(X_log, y_log)
# xTree2 = regModel(X_log, y_log)
# print(features2.chiSquareTest(10))
# print(xTree2.extraTree(10))

# Feature selection is done lets see the model
r = regModel(X_linreg, y)
# print(r.reg())  # I'm getting ValueError: x and y must be the same size
# looks like judging from the p-Values we can get rid of a few, Exterior 2nd, Total rooms above ground, OpenPorch SF,
# Year Build, Month Sold (should have extracted that earlier how embarrassing)
# Neighborhood is borderline, interesting
# After removing the above explanatory variables, the model improved tremendously
#
# Now lets try splitting the data, see above



