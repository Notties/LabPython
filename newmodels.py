import pandas as pd
import numpy as np
import pythainlp
import re
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import thai_stopwords
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, Conv1D, MaxPooling1D, Dropout, LSTM
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.models import load_model


# nltk.download('punkt')

# Load the English data from CSV file
df_en = pd.read_csv("datasetEN.csv", encoding='utf-8')

# Load the Thai data from CSV file
df_th = pd.read_csv("datasetTH.csv", encoding='utf-8')

def preprocess_text(text, lang):
    if lang == "th":
        # tokenize the text
        tokens = pythainlp.word_tokenize(str(text), engine='newmm')
        # remove stop words
        stop_words = thai_stopwords()
        tokens = [word for word in tokens if word not in stop_words]
    else: # lang == "en"
        # tokenize the text
        tokens = nltk.word_tokenize(str(text))
        # remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        # stem the words
        stemmer = SnowballStemmer('english')
        tokens = [stemmer.stem(word) for word in tokens]
        
    # join the tokens back into a single string
    text = " ".join(tokens)
    # remove non-alphabetic characters and extra whitespaces
    text = re.sub('[^A-Za-zก-๙]+', ' ', text).strip()
    return text

# Preprocess the English text data
df_en['text'] = df_en['text'].apply(preprocess_text, lang="en")

# Preprocess the Thai text data
df_th['text'] = df_th['text'].apply(preprocess_text, lang="th")

# Combine the preprocessed English and Thai data into a single DataFrame
df = pd.concat([df_en, df_th], ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# save X_train to a binary file
np.save('X_train.npy', X_train)

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# vocab_size = len(tokenizer.word_index) + 1

# Pad the sequences to a maximum length of MAX_SEQUENCE_LENGTH
MAX_SEQUENCE_LENGTH = 100
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

# Convert the sentiment labels to one-hot encoded vectors
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# Define the CNN model
num_words=10000
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, callbacks=[es])

# Save the model to a file
model.save('sentiment_analysis_model.h5')

# Test the model on new data
text = preprocess_text("This is a test sentence.", lang="en")
seq = tokenizer.texts_to_sequences([text])
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = ['Negative', 'Neutral', 'Positive']
print('Sentiment:', labels[np.argmax(pred)])