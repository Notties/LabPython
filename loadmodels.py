import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Load the saved trained model
model = load_model('sentiment_model.h5')

# Preprocess the input text using the same tokenizer used during training
csv = pd.read_csv("data.csv", encoding='utf-8')
tokenizer = Tokenizer(num_words=5000, split=' ')
tokenizer.fit_on_texts(csv['text'].values)

data = pd.concat([csv], ignore_index=True)

text = 'สินค้าไม่มีคุณภาพ..'
seq = tokenizer.texts_to_sequences([text])
padded = pad_sequences(seq, 9)
pred = model.predict(padded)
labels = ['Negative', 'Neutral', 'Positive']
print('Sentiment:', labels[np.argmax(pred)], ', Percentage:', round(pred[0][np.argmax(pred)] * 100, 2), '%')
