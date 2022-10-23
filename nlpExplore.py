import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import seaborn as sn
import sklearn
import gensim
import pyLDAvis
import wordcloud
import textblob
import spacy
import textstat

cTrain = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/cola_train.csv')
print(cTrain.shape)
print(cTrain.columns)
#
from nltk.tokenize import TweetTokenizer
tk = TweetTokenizer()
cTrain['tokens_raw'] = cTrain['text'].apply(lambda x: tk.tokenize(x.lower()))
print(cTrain.head())


