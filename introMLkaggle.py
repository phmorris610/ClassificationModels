"""Start with the decision tree, breaks the data
into groups, capturing patterns from data is called
filling or training the model. The data used to fit the
model is called the training data. After the data has
been fit, we can apply it to new data to predict

Using more categories in you model is referred to as
using more "splits", called a deeper tree

The prediction is at the bottom of the tree, the place
where we make a prediction is called a leaf
"""
import pandas as pd
import numpy as np

data = pd.read_csv("../../Desktop/DeepLearn/mHousing.csv")  # use the backslash
# print(data.describe())
# print(data)
# print(data.columns)
# print(data.size)
data = data.dropna()
# print(data.size)
y = data.Price
# print(round(y.describe(), 2))
melFeat = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[melFeat]
# print(X.describe())
# print(X.head())
X.dropna()
from sklearn.tree import DecisionTreeRegressor

melModel = DecisionTreeRegressor(random_state=1)
melModel.fit(X, y)
# print(X.isna().sum())
# print(data.head(10))
# print(y.head(10))
# print(melModel.predict(X.head(10)))
# out first loss function is MAE, Mean Absolute Error
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melModel.predict(X)
print(mean_absolute_error(y, predicted_home_prices))
# we have just used the same data to both model and evaluate, the evaluation metric is called an in-sample score
# testing the model on data that it hasn't seen before is called validation data
# to split the data we use train_test_split from scikit-learn
# some will be use for training, some for validation
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
melModel = DecisionTreeRegressor(random_state=1)  # Define model
melModel.fit(train_X, train_y)  # get predicted prices on validation data
val_predictions = melModel.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))  # get predicted prices on validation data

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))