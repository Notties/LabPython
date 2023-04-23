from keras.models import load_model
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
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Load the saved model
model = load_model('sentiment_analysis_model.h5')

# load X_train from the saved binary file
X_train = np.load('X_train.npy', allow_pickle=True)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Define the preprocess_text function
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

# Define the function to preprocess the new data and make predictions
def predict_sentiment(text, lang):
    # Preprocess the text
    text = preprocess_text(text, lang)
    # Tokenize the text
    text = tokenizer.texts_to_sequences([text])
    # Pad the sequences to a maximum length of MAX_SEQUENCE_LENGTH
    text = pad_sequences(text, maxlen=100)
    # Make predictions
    predictions = model.predict(text)
    # Return the sentiment label and percentage
    sentiment_labels = ['negative', 'neutral', 'positive']
    sentiment = sentiment_labels[np.argmax(predictions)]
    percentage = round(np.max(predictions)*100, 2)
    return sentiment, percentage


# text = "It's good"
# sentiment, percentage = predict_sentiment(text, lang="en")
# print(sentiment, percentage)

# text = "It's bad"
# sentiment, percentage = predict_sentiment(text, lang="en")
# print(sentiment, percentage)

text = "แย่มากเลยนะ ไม่คิดว่าจะ"
sentiment, percentage = predict_sentiment(text, lang="th")
print(sentiment, percentage)

# text = "ดีมาก"
# sentiment, percentage = predict_sentiment(text, lang="th")
# print(sentiment, percentage)