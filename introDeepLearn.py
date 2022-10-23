
import pandas as pd

red_wine = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/wine.csv', sep=';', index_col=False)

print(red_wine.head())
X = red_wine.drop(['quality'], axis=1)
y = red_wine.pop('quality')
print(X.head())
print(y.head())