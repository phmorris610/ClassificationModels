from sklearn.svm._libsvm import predict_proba


class Classification:
    def __init__(self, X, y):
        self.X = X
        self.y = y

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

    def normal(self, feature):
        import matplotlib.pyplot as plt
        plt.hist(self.X[feature], bins=20, edgecolor='black')
        plt.show()

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


class DecisionTree(Classification):
    def __init__(self, X, y):
        super().__init__(X, y)

    def decision(self, n, feat):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import tree
        import matplotlib.pyplot as plt
        clf = tree.DecisionTreeClassifier(max_depth=n)  # set hyperparameter
        clf.fit(self.X, self.y)
        plt.figure(figsize=(12, 12))  # set plot size (denoted in inches)
        tree.plot_tree(clf, feature_names=feat, class_names=True, filled=True, fontsize=10)
        plt.show()


class RandomForest(Classification):
    def __init__(self, X, y):
        super().__init__(X, y)

    def rForestClass(self, n):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn import metrics
        clf = RandomForestClassifier(n_estimators=n)
        clf.fit(self.X, self.y)
        y_pred = clf.predict(self.X)
        print("Accuracy: ", metrics.accuracy_score(self.y, y_pred))

    def rForestReg(self, n, state):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn import metrics
        regr = RandomForestRegressor(max_depth=n, random_state=state)
        regr.fit(self.X, self.y)
        y_pred = regr.predict(self.X)
        print("Accuracy: ", metrics.accuracy_score(self.y, y_pred))


class NiaveBayes(Classification):
    def __init__(self, X, y):
        super().__init__(X, y)

    def gBayes(self):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score, r2_score
        gnb = GaussianNB()
        gnb.fit(self.X, self.y)
        y_pred = gnb.predict(self.X)
        print('Accuracy Score, R^2')
        return accuracy_score(self.y, gnb.predict(self.X), sample_weight=None), r2_score(self.y, y_pred,
                                                                                         sample_weight=None)

    def mBayes(self):
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import accuracy_score, r2_score
        mnb = MultinomialNB()
        mnb.fit(self.X, self.y)
        y_pred     = mnb.predict(self.X)
        print('Accuracy Score, R^2')
        return accuracy_score(self.y, mnb.predict(self.X), sample_weight=None), r2_score(self.y, y_pred,
                                                                                         sample_weight=None)

    def compBayes(self):
        from sklearn.naive_bayes import ComplementNB
        from sklearn.metrics import accuracy_score, r2_score
        cnb = ComplementNB()
        cnb.fit(self.X, self.y)
        y_pred = cnb.predict(self.X)
        print('Accuracy Score, R^2')
        return accuracy_score(self.y, cnb.predict(self.X), sample_weight=None), r2_score(self.y, y_pred,
                                                                                         sample_weight=None)

    def bBayes(self):
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.metrics import accuracy_score, r2_score
        bnb = BernoulliNB()
        bnb.fit(self.X, self.y)
        y_pred = bnb.predict(self.X)
        print('Accuracy Score, R^2')
        return accuracy_score(self.y, bnb.predict(self.X), sample_weight=None), r2_score(self.y, y_pred,
                                                                                         sample_weight=None)

    def catBayes(self):
        from sklearn.naive_bayes import CategoricalNB
        from sklearn.metrics import accuracy_score, r2_score
        cnb = CategoricalNB()
        cnb.fit(self.X, self.y)
        y_pred = cnb.predict(self.X)
        print('Accuracy Score, R^2')
        return accuracy_score(self.y, cnb.predict(self.X), sample_weight=None), r2_score(self.y, y_pred,
                                                                                         sample_weight=None)







# Import and wrangle the Titanic Dataset
import pandas as pd
import numpy as np

train_df = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/titanic3.csv')
# Look at your data
# print("train_df.shape: ", train_df.shape)
# print(train_df.columns)
# print(train_df.describe())
null_entries = train_df.isnull().sum()
# print(null_entries)
# With practically all cabin columns being zero we can safely remove them, plus it's safe to say if you were wealthy
# you were probably traveling first class which will show in ticket and fare
train_df.drop(columns=["Cabin"], inplace=True)
# Age and embarked have missing values, age seems important so lets remove rows with missing Age/Embarked values
# then reset the index
train_df.dropna(axis=0, inplace=True)
train_df.reset_index(inplace=True)
# check shape of data after removing those rows/cols, we went from 891 to 712 rows, not bad
# print(train_df.shape)
X = train_df.copy()  # X values, aka observations, explanatory variables, independent variables
y = X.pop("Survived")  # the y value, aka the response, the dependent variable
# Now let's encode the categoricals
for colname in X.select_dtypes('object').columns:
    X[colname], _ = X[colname].factorize()
# Now the data is set, and it's clean, lets select some features
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
f = Classification(X_train, y_train)
print(f.extraTree(12))
# We can see sex and age are the most important features, lets make a list, we can remove passID and index
featured_selections = ['Fare', 'Age', 'Sex', 'Ticket']
X_feat = X_train[featured_selections].copy()
X_test = X_test[featured_selections].copy()
feature_names = X.columns
labels = y.unique()
feature_names2 = X_feat.columns
f1 = Classification(X_feat, y)
f2 = Classification(X_test, y_test)
# print(f1.extraTree(10))
# We can see removing these variable actually resulted in a stronger relation from these to the y
# Lets quick check if the data is normal
# print(f.normal('Sex'))
# not exactly, lets see the regression output
# print(f1.reg())
# The residuals look great, but judging from those p-Values we can remove Name at least, lets do that then see
# Well removing those variable really tossed a wrench in but improved the R^2 greatly, lets remove SibSp
# The residuals are still symmetric but now Age Pclass and Embarked have very high p-values, lets remove pclass/embarked
# With or without the variable Age the R^2 and MSE remain steady so this is puzzling the p-Value of age remains high
# I will keep it because the residuals with Age look far nicer
#
# Let's check out a decision tree
t = DecisionTree(X, y)
# print(t.decision(3, feature_names))
r = RandomForest(X_feat, y)
# print(r.rForestClass(50))
# Lets try the Niave Bayes
b = NiaveBayes(X_test, y_test)  # Using the test data the Niave Bayes Categorical actually has a higher accuracy 91%!
# print(b.compBayes())
# print(b.gBayes())
# print(b.mBayes())
print(b.catBayes())
# print(b.bBayes())





