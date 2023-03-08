import tensorflow as tf
Model = tf.keras.models.Model
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
load_model = tf.keras.models.load_model

import pandas as pd
import re
from pythainlp.tokenize import word_tokenize, Tokenizer
KRTokenizer = tf.keras.preprocessing.text.Tokenizer

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import numpy as np

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional, Embedding, Dropout, BatchNormalization
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.keras.optimizers import Adam

#import seaborn as sn
import matplotlib.pyplot as plt

import pickle as p
#import plotly
#import plotly.graph_objs as go

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import string
# from os import listdir
from string import punctuation
# from os import listdir

#########################
from pythainlp.tokenize import word_tokenize #, Tokenizer
from pythainlp.corpus.common import thai_words

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from pythainlp.corpus import thai_stopwords

#from gensim.models import Word2Vec

EPOCHS = 100
BS = 32
DIMENSION = 100

comments = []
labels = []
csv = pd.read_csv("data.csv", encoding='utf-8')
texts = csv['text'].tolist()
sentiments = csv['sentiment'].tolist()
        
df = pd.DataFrame({ "sentiments": sentiments, "texts": texts })
df = df.drop_duplicates()

neu_df = df[df.sentiments == "neutral"]

pos_df = df[df.sentiments == "positive"]

neg_df = df[df.sentiments == "negative"]

sentiment_df = pd.concat([neg_df, pos_df])

sentiment_df['clean_comments'] = sentiment_df['texts'].fillna('').apply(lambda x: x.lower())

pun = '"#\'()*,-.;<=>[\\]^_`{|}~'

sentiment_df['clean_comments'] = sentiment_df['clean_comments'].str.replace(r'[%s]' % (pun), '', regex=True)

custom_words_list = set(thai_words())

text = "โอเคบ่พวกเรารักภาษาบ้านเกิด"
custom_tokenizer = Tokenizer(custom_words_list)
custom_tokenizer.word_tokenize(text)

sentiment_df['clean_comments'] = sentiment_df['clean_comments'].apply(lambda x: custom_tokenizer.word_tokenize(x))

sentiment_df.sample(5)