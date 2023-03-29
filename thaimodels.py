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

import seaborn as sn
import matplotlib.pyplot as plt

import pickle as p
import plotly
import plotly.graph_objs as go

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

from gensim.models import Word2Vec


EPOCHS = 100
BS = 32
DIMENSION = 100

comments = []
labels = []


with open("train.txt",encoding="utf-8") as f:
    for line in f:
        comments.append(line.strip())

with open("train_label.txt",encoding="utf-8") as f:
    for line in f:
        labels.append(line.strip())
        
        
df = pd.DataFrame({ "category": labels, "comments": comments })
df.head()


df = df.drop_duplicates()


neu_df = df[df.category == "neu"].sample(4300)
neu_df.head()


pos_df = df[df.category == "pos"]
pos_df.head()


neg_df = df[df.category == "neg"].sample(4300)
neg_df.head()


sentiment_df = pd.concat([neg_df, pos_df])
sentiment_df.head()


sentiment_df['clean_comments'] = sentiment_df['comments'].fillna('').apply(lambda x: x.lower())
sentiment_df.head()


pun = '"#\'()*,-.;<=>[\\]^_`{|}~'
pun


sentiment_df['clean_comments'] = sentiment_df['clean_comments'].str.replace(r'[%s]' % (pun), '', regex=True)


sentiment_df.sample(5)


custom_words_list = set(thai_words())
len(custom_words_list)


text = "โอเคบ่พวกเรารักภาษาบ้านเกิด"
custom_tokenizer = Tokenizer(custom_words_list)
custom_tokenizer.word_tokenize(text)


sentiment_df['clean_comments'] = sentiment_df['clean_comments'].apply(lambda x: custom_tokenizer.word_tokenize(x))


sentiment_df.sample(5)


tokenized_doc = sentiment_df['clean_comments']
tokenized_doc[:10]


tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])


tokenized_doc[:10]


tokenized_doc = tokenized_doc.to_list()


# de-tokenization
detokenized_doc = []
for i in range(len(tokenized_doc)):
#     print(tokenized_doc[i])
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
sentiment_df['clean_comments'] = detokenized_doc


sentiment_df.head()


cleaned_words = sentiment_df['clean_comments'].to_list()


cleaned_words[:1]


def create_tokenizer(words, filters = ''):
    token = KRTokenizer()
    token.fit_on_texts(words)
    return token


train_word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(train_word_tokenizer.word_index) + 1

train_word_tokenizer.word_index


def max_length(words):
    return(len(max(words, key = len)))


max_length = max_length(tokenized_doc)
max_length


def encoding_doc(token, words):
    return(token.texts_to_sequences(words))


encoded_doc = encoding_doc(train_word_tokenizer, cleaned_words)


print(cleaned_words[0])
print(encoded_doc[0])


def padding_doc(encoded_doc, max_length):
   return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


padded_doc = padding_doc(encoded_doc, max_length)
print("Shape of padded docs = ",padded_doc.shape)

print(cleaned_words[0])
print(encoded_doc[0])
print(padded_doc[0])


category = sentiment_df['category'].to_list()


unique_category = list(set(category))
unique_category


output_tokenizer = create_tokenizer(unique_category)


encoded_output = encoding_doc(output_tokenizer, category)
print(category[0:2])
print(encoded_output[0:2])


encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
encoded_output.shape


def one_hot(encode):
  oh = OneHotEncoder(sparse = False)
  return(oh.fit_transform(encode))


output_one_hot = one_hot(encoded_output)
print(encoded_output[0])
print(output_one_hot[0])


train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2, stratify=output_one_hot)


print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


num_classes = len(unique_category)


from tensorflow.keras.optimizers import Adam
adam = Adam(learning_rate=0.0001)


# define the model
def define_model(length, vocab_size):
    # channel 1
    inputs1 = tf.keras.layers.Input(shape=(length,))
    embedding1 = tf.keras.layers.Embedding(vocab_size, DIMENSION, trainable = True)(inputs1)
    conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = tf.keras.layers.Dropout(0.5)(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
    flat1 = tf.keras.layers.Flatten()(pool1)
    # channel 2
    inputs2 = tf.keras.layers.Input(shape=(length,))
    embedding2 = tf.keras.layers.Embedding(vocab_size, DIMENSION, trainable = True)(inputs2)
    conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = tf.keras.layers.Dropout(0.5)(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop2)
    flat2 = tf.keras.layers.Flatten()(pool2)
    # channel 3
    inputs3 = tf.keras.layers.Input(shape=(length,))
    embedding3 = tf.keras.layers.Embedding(vocab_size, DIMENSION, trainable = True)(inputs3)
    conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
    drop3 = tf.keras.layers.Dropout(0.5)(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop3)
    flat3 = tf.keras.layers.Flatten()(pool3)
    # merge
    merged = tf.keras.layers.concatenate([flat1, flat2, flat3])
    # interpretation
    dense1 = tf.keras.layers.Dense(10, activation='relu')(merged)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    # compile
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    # summarize
    print(model.summary())
#     plot_model(model, show_shapes=True, to_file='multichannel.png')
    return model


model = define_model(max_length, vocab_size)


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 3, verbose=1,factor=0.1, min_lr=0.000001)
callbacks_list = [checkpoint, learning_rate_reduction]


hist = model.fit([train_X, train_X, train_X], train_Y, epochs = EPOCHS, batch_size = BS, validation_data = ([val_X, val_X, val_X], val_Y), callbacks = [callbacks_list], shuffle=True)


h1 = go.Scatter(y=hist.history['loss'], 
                    mode="lines", line=dict(
                    width=2,
                    color='blue'),
                    name="loss"
                   )
h2 = go.Scatter(y=hist.history['val_loss'], 
                    mode="lines", line=dict(
                    width=2,
                    color='red'),
                    name="val_loss"
                   )

data = [h1,h2]
layout1 = go.Layout(title='Loss',
                   xaxis=dict(title='epochs'),
                   yaxis=dict(title=''))
fig1 = go.Figure(data = data, layout=layout1)
fig1.show()


h1 = go.Scatter(y=hist.history['accuracy'], 
                    mode="lines", line=dict(
                    width=2,
                    color='blue'),
                    name="acc"
                   )
h2 = go.Scatter(y=hist.history['val_accuracy'], 
                    mode="lines", line=dict(
                    width=2,
                    color='red'),
                    name="val_acc"
                   )

data = [h1,h2]
layout1 = go.Layout(title='Accuracy',
                   xaxis=dict(title='epochs'),
                   yaxis=dict(title=''))
fig1 = go.Figure(data = data, layout=layout1)
fig1.show()


predict_model = load_model(filename) 
predict_model.summary()


score = predict_model.evaluate([val_X, val_X, val_X], val_Y, verbose=0)
print('Validate loss:', score[0])
print('Validate accuracy:', score[1])


y_true = np.argmax(val_Y,axis = 1)
print(val_Y[0])
print(y_true[0])


cm = confusion_matrix(y_true, predicted_classes)
np.savetxt("confusion_matrix.csv", cm, delimiter=",")


df_cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize=(20,14))
sn.set(font_scale=1.2) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt='g') # for num predict size

plt.show()


label = [key for key, value in label_dict.items()]


print(classification_report(y_true, predicted_classes, target_names=label, digits=4))