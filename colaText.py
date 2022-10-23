import pandas as pd
import numpy as np

cTrain = pd.read_csv('C:/Users/Paul Morris/Desktop/DeepLearn/cola_train.csv')
print(cTrain.shape)
print(cTrain.columns)
#
from nltk.tokenize import TweetTokenizer
tk = TweetTokenizer()
cTrain['tokens_raw'] = cTrain['text'].apply(lambda x: tk.tokenize(x.lower()))
print(cTrain.head())

import re
from nltk.corpus import stopwords
# nltk.download('stopwords')
# stops = set(stopwords.words('english'))
chars2remove = set(['.', '!', '/', '?'])
# cTrain['tokens_raw'] = cTrain['tokens_raw'].apply(lambda x: [w for w in x if w not in stops])
cTrain['tokens_raw'] = cTrain['tokens_raw'].apply(lambda x: [w for w in x if w not in chars2remove])
print(cTrain.head())
#

# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# cTrain['tokens'] = cTrain['tokens_raw'].apply(lambda x: [lemmatizer.lemmatize(w, pos="v") for w in x])
# #df['tokens'] = df['tokens_raw'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
# print(cTrain.head())






