import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Dropout

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# load the data from CSV file
csv = pd.read_csv("data.csv", encoding='utf-8')
texts = csv['text'].tolist()
sentiments = csv['sentiment'].tolist()

# th_data = pd.read_csv('th_dataset.csv', encoding='utf-8')
# en_data = pd.read_csv('en_dataset.csv', encoding='utf-8')

data = pd.concat([csv], ignore_index=True)


# prepare the text data
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

Y = pd.get_dummies(data['sentiment']).values

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# build the model
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# train the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=40, batch_size=128)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='best')
plt.show()

# save the model
model.save("sentiment_model.h5")

# load the model
loaded_model = load_model("sentiment_model.h5")

print("X.shape[1]: ",X.shape[1])

text = 'สินค้าไม่มีคุณภาพ'
seq = tokenizer.texts_to_sequences([text])
padded = pad_sequences(seq, maxlen=X.shape[1])
pred = model.predict(padded)
labels = ['Negative', 'Neutral', 'Positive']
print('Sentiment:', labels[np.argmax(pred)], ', Percentage:', round(pred[0][np.argmax(pred)] * 100, 2), '%')