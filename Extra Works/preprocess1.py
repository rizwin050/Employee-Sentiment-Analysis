import re
import unicodedata
import numpy as np
import nltk

from nltk.corpus import stopwords, opinion_lexicon, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import lightgbm

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('opinion_lexicon')

# Load models
with open('model/lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 200  # same as training setup

# Preprocessing function
def preprocess_text(sen):
    sentence = unicodedata.normalize('NFKD', sen)
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_words = [word for word in words if word not in stop_words]
    sentence = ' '.join(filtered_words)
    return sentence

# Get part of speech tag
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)

# Lemmatization
def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)) for word in tokens]
    return lemmas

# Predict sentiment
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    tokens = word_tokenize(processed_review)
    lemmatized_text = lemmatize_text(tokens)
    processed_text = ' '.join(lemmatized_text)

    description_length = len(processed_text)
    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    positive_word_count = len([word for word in processed_text.split() if word in positive_words])
    negative_word_count = len([word for word in processed_text.split() if word in negative_words])

    #debug
    print("input : ",review)
    print("sentence after cleaning : ",processed_review )
    print("sentence after processed text : ",processed_text)
    print("pos word count : ",positive_word_count)
    print("negative word count : ",negative_word_count )

    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Combine features (update if you include more features)
    additional_features = np.array([[positive_word_count,negative_word_count,description_length]])
    combined_features = np.hstack((padded_sequence, additional_features))

    sentiment = model.predict(combined_features)
    print(f"Raw model output : {sentiment}")
    return sentiment