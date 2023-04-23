import numpy as np
import pandas as pd
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
from keras.models import load_model


# Load the saved model
model = load_model('sentiment_analysis_model.h5')

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

# Define the labels for sentiment categories
labels = ['Negative', 'Neutral', 'Positive']

# Load X_train from the CSV file
X_train = pd.read_csv('X_train.csv')

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Define a function to predict the sentiment of a text
def predict_sentiment(text, lang):
    # Preprocess the text data
    text = preprocess_text(text, lang)

    # Convert the text to a sequence of integers
    sequence = tokenizer.texts_to_sequences([text])

    # Pad the sequence with zeros to match the max sequence length
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Use the model to predict the sentiment category of the text
    pred = model.predict(padded_sequence)[0]

    # Get the index of the predicted category with the highest probability
    index = np.argmax(pred)

    # Return the predicted sentiment category and the probability of the prediction
    return labels[index], pred[index]

# Test the model on new data
text = 'I love this product!'
sentiment, prob = predict_sentiment(text, lang="en")
print(f'Text: {text}')
print(f'Sentiment: {sentiment}, Probability: {prob}')

text = 'ก็ดีนะ'
sentiment, prob = predict_sentiment(text, lang="th")
print(f'Text: {text}')
print(f'Sentiment: {sentiment}, Probability: {prob}')