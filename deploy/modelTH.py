import re
import pythainlp
import emoji
import pickle
import numpy as np

from pythainlp.corpus.common import thai_stopwords
from keras.models import load_model
from keras.utils import pad_sequences

#load model and tokenizer
with open('C:/Sentiment-Analysis-Model/savedmodel/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('C:/Sentiment-Analysis-Model/savedmodel/sentiment_analysis_model.h5')

# Define preprocess for Thai text
def preprocess_text(text):
    # tokenize the text
    text = emoji.demojize(text)
    text = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    text = re.sub(r'[a-zA-Z]', '', text)  # Remove English characters
    text = " ".join(word for word in text)
    text = "".join(word for word in text.split() if word.lower() not in thai_stopwords())
    tokens = pythainlp.word_tokenize(str(text), engine='newmm')
    # join the tokens back into a single string
    text = " ".join(tokens)
    # remove non-alphabetic characters and extra whitespaces
    text = re.sub('[^A-Za-zก-๙]+', ' ', text).strip()
    return text

def predict(new_text):
    new_text = preprocess_text(new_text)

    new_text = tokenizer.texts_to_sequences([new_text])  # Convert text to sequences of integers
    new_text = pad_sequences(new_text, maxlen=128)
    # Make the prediction
    prediction = model.predict(new_text)[0]

    # Get the predicted sentiment and confidence level
    sentiments = ['Negative', 'Neutral', 'Positive']
    sentiment = sentiments[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)), 2)

    # Display the result
    return sentiment, confidence
