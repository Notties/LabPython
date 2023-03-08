# import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# load the data from CSV file
data = pd.read_csv('data.csv')

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

# preprocess the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# create the Keras embedding layer
embedding_dim = 100
filters = 250
kernel_size = 3
hidden_dims = 250

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='best')
plt.show()

# save the model
model.save('sentiment_model.h5')

# load the model and use it to predict sentiment
loaded_model = load_model('sentiment_model.h5')
new_text = "This movie is amazing!"
new_text = tokenizer.texts_to_sequences([new_text])
new_text = pad_sequences(new_text, padding='post', maxlen=maxlen)
prediction = loaded_model.predict(new_text)
if prediction > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")
