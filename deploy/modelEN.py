import numpy as np
import nltk
import pickle
import re
import emoji

from nltk.corpus import stopwords
from keras.models import load_model
from keras.utils import pad_sequences

#load model and tokenizer
with open('C:/Sentiment-Analysis-Model/savedmodel/tokenizerEN.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('C:/Sentiment-Analysis-Model/savedmodel/sentiment_analysis_modelEN.h5')

# Preprocess the text
def preprocess_text(text):
    # Check if text is not null
    if isinstance(text, str):
        text = emoji.demojize(text)
        # Remove special characters and convert to lowercase
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back into a string
        text = " ".join(tokens)
    return text

def predictEN(new_text):
    new_text = preprocess_text(new_text)
    new_text = tokenizer.texts_to_sequences([new_text])  # Convert text to sequences of integers
    new_text = pad_sequences(new_text, maxlen=128)
    # Make the prediction
    prediction = model.predict(new_text)[0]

    # Get the predicted sentiment and confidence level
    sentiments = ['negative', 'neutral', 'positive']
    sentiment = sentiments[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)), 2)
    percent = round(confidence * 100)

    return {'sentiment': sentiment, 'percentage': f'{percent}'}

def predictTextObjectEN(new_text):
    new_text = preprocess_text(new_text)
    new_text = tokenizer.texts_to_sequences([new_text])  # Convert text to sequences of integers
    new_text = pad_sequences(new_text, maxlen=128)
    # Make the prediction
    prediction = model.predict(new_text)[0]

    # Get the predicted sentiment and confidence level
    sentiments = ['negative', 'neutral', 'positive']
    sentiment = sentiments[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)), 2)
    percent = round(confidence * 100)

    return sentiment, f'{percent}'