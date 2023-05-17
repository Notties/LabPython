import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

from pythainlp.tokenize import word_tokenize, Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from pythainlp.corpus.common import thai_words
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from gensim.models import Word2Vec

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
KRTokenizer = tf.keras.preprocessing.text.Tokenizer
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
Model = tf.keras.models.Model


# Load the Thai data from CSV file
df_th = pd.read_csv("C:/LabPython/datasets/dataTH.csv", encoding='utf-8')
df = df_th.drop_duplicates()

neg_df = df[df.sentiment == "negative"]
pos_df = df[df.sentiment == "positive"]
neu_df = df[df.sentiment == "neutral"]
sentiment_df = pd.concat([neg_df, pos_df, neu_df])

sentiment_df['clean_comments'] = df_th['text'].fillna('').apply(lambda x: x.lower())
pun = '"#\'()*,-.;<=>[\\]^_`{|}~'
pun
sentiment_df['clean_comments'] = sentiment_df['clean_comments'].str.replace(r'[%s]' % (pun), '', regex=True)
custom_words_list = set(thai_words())
print('custom_words_list', len(custom_words_list))

text = "โอเคบ่พวกเรารักภาษาบ้านเกิด"
custom_tokenizer = Tokenizer(custom_words_list)
print('word_tokenize', custom_tokenizer.word_tokenize(text))

sentiment_df['clean_comments'] = sentiment_df['clean_comments'].apply(lambda x: custom_tokenizer.word_tokenize(x))

tokenized_doc = sentiment_df['clean_comments']
tokenized_doc = tokenized_doc.to_list()
with open('tokenized_doc.pkl', 'wb') as file:
    pickle.dump(tokenized_doc, file)
    
# de-tokenization
detokenized_doc = []
for i in range(len(tokenized_doc)):
#     print(tokenized_doc[i])
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
    
sentiment_df['clean_comments'] = detokenized_doc

cleaned_words = sentiment_df['clean_comments'].to_list()
print('cleaned_words', cleaned_words[:1])

with open('cleaned_words.pkl', 'wb') as file:
    pickle.dump(cleaned_words, file)

def create_tokenizer(words, filters = ''):
    token = KRTokenizer()
    token.fit_on_texts(words)
    return token

train_word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(train_word_tokenizer.word_index) + 1
print('train_word_tokenizer', train_word_tokenizer)

with open('train_word_tokenizer.pkl', 'wb') as file:
    pickle.dump(train_word_tokenizer, file)

def max_length(words):
    return(len(max(words, key = len)))

max_length = max_length(tokenized_doc)
print('max_length', max_length)

def encoding_doc(token, words):
    return(token.texts_to_sequences(words))

encoded_doc = encoding_doc(train_word_tokenizer, cleaned_words)

print('cleaned_words ',cleaned_words[0])
print('encoded_doc ',encoded_doc[0])

def padding_doc(encoded_doc, max_length):
   return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

padded_doc = padding_doc(encoded_doc, max_length)
print("Shape of padded docs = ",padded_doc.shape)

category = sentiment_df['sentiment'].to_list()
unique_category = list(set(category))
print('unique_category ', unique_category)

output_tokenizer = create_tokenizer(unique_category)
print('output_tokenizer', output_tokenizer)

with open('output_tokenizer.pkl', 'wb') as file:
    pickle.dump(output_tokenizer, file)

encoded_output = encoding_doc(output_tokenizer, category)
print('category[0:2] ',category[0:2])
print('encoded_output[0:2]', encoded_output[0:2])

encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)
print('encoded_output.shape ', encoded_output.shape)


def one_hot(encode):
  oh = OneHotEncoder(sparse_output = False)
  return(oh.fit_transform(encode))

num_classes = len(unique_category)
output_one_hot = one_hot(encoded_output)
print('encoded_output[0] ', encoded_output[0])
print('output_one_hot[0] ', output_one_hot[0])

train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2, stratify=output_one_hot)

print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

# define the model
adam = Adam(learning_rate=0.0001)
EPOCHS = 40
BS = 32
DIMENSION = 100

sentences = [st.split() for st in cleaned_words]
w2v_model = Word2Vec(sentences, min_count=1, vector_size=DIMENSION, workers=6, sg=1, epochs=500)

w2v_model.save('w2v_model.bin')
new_model = Word2Vec.load('w2v_model.bin')

embedding_matrix = np.zeros((vocab_size, DIMENSION))

for word, i in train_word_tokenizer.word_index.items():
    if word in new_model.wv.key_to_index:
        embedding_vector = new_model.wv[word]
        embedding_matrix[i] = embedding_vector

# define the model
def define_w2v_model(length, vocab_size, embedding_matrix):
    # channel 1
    inputs1 = tf.keras.layers.Input(shape=(length,))
    embedding1 = tf.keras.layers.Embedding(vocab_size, DIMENSION, trainable = False, weights=[embedding_matrix])(inputs1)
    conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = tf.keras.layers.Dropout(0.5)(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop1)
    flat1 = tf.keras.layers.Flatten()(pool1)
    # channel 2
    inputs2 = tf.keras.layers.Input(shape=(length,))
    embedding2 = tf.keras.layers.Embedding(vocab_size, DIMENSION, trainable = False, weights=[embedding_matrix])(inputs2)
    conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
    drop2 = tf.keras.layers.Dropout(0.5)(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop2)
    flat2 = tf.keras.layers.Flatten()(pool2)
    # channel 3
    inputs3 = tf.keras.layers.Input(shape=(length,))
    embedding3 = tf.keras.layers.Embedding(vocab_size, DIMENSION, trainable = False, weights=[embedding_matrix])(inputs3)
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

model2 = define_w2v_model(max_length, vocab_size, embedding_matrix)

filename = 'model2.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 3, verbose=1,factor=0.1, min_lr=0.000001)
callbacks_list = [checkpoint, learning_rate_reduction]

hist2 = model2.fit([train_X, train_X, train_X], train_Y, epochs = EPOCHS, batch_size = BS, validation_data = ([val_X, val_X, val_X], val_Y), callbacks = [callbacks_list], shuffle=True)

# Save the model to a file
predict_model = load_model(filename) 
score = predict_model.evaluate([val_X, val_X, val_X], val_Y, verbose=0)
print('Validate loss:', score[0])
print('Validate accuracy:', score[1])

predicted_classes = np.argmax(predict_model.predict([val_X, val_X, val_X]), axis=-1)
print('predicted_classes', predicted_classes.shape)

y_true = np.argmax(val_Y,axis = 1)
print(val_Y[0])
print(y_true[0])

label_dict = output_tokenizer.word_index
label = [key for key, value in label_dict.items()]
print(classification_report(y_true, predicted_classes, target_names=label, digits=4))

cm = confusion_matrix(y_true, predicted_classes)
np.savetxt("confusion_matrix.csv", cm, delimiter=",")

df_cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize=(20,14))
sn.set(font_scale=1.2) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt='g') # for num predict size

plt.show()