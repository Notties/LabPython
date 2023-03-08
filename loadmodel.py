# import necessary libraries
from keras.models import load_model
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('data.csv')

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2)

# load the tokenizer used to preprocess the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# load the saved Keras model
model = load_model('sentiment_model.h5')

# preprocess the new text data
maxlen = 100
new_text = "The service at this restaurant was terrible, I would not recommend it."
new_text = tokenizer.texts_to_sequences([new_text])
new_text = pad_sequences(new_text, padding='post', maxlen=maxlen)

prediction = model.predict(new_text)
# convert the prediction to percentages
positive_percentage = prediction[0][0] * 100
negative_percentage = (1 - prediction[0][0]) * 100

# output the percentages
print("Positive sentiment: {:.2f}%".format(positive_percentage))
print("Negative sentiment: {:.2f}%".format(negative_percentage))

if np.squeeze(prediction) > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")
