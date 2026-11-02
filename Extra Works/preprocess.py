import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
import emoji
import unicodedata
import numpy as np
import nltk
import pickle
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('opinion_lexicon')
from nltk.corpus import stopwords, opinion_lexicon
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import joblib


max_length=200

def remove_strings(text, strings_to_remove):
    for string in strings_to_remove:
        text = text.replace(string, "")
    return text

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
    sentence = emoji.replace_emoji(sentence, replace='')
    return sentence

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)) for word in tokens]
    return lemmas

def categorize_rating(rating):
    if rating >= 3.5:
        return 2  # Positive
    elif rating <= 2:
        return 0  # Negative
    else:
        return 1  # Neutral

def preprocess_data(input_data):
    strings_to_remove = [
        "What is the best part of working at the company?",
        "What is the most stressful part about working at the company?",
        "What is the work environment and culture like at the company?",
        "What is a typical day like for you at the company?",
        "\n",
    ]

    if isinstance(input_data, str) and not input_data.endswith('.csv'):
        processed_review = preprocess_text(input_data)
        tokens = word_tokenize(processed_review)
        lemmatized_text = lemmatize_text(tokens)
        processed_text = ' '.join(lemmatized_text)

        description_length = len(processed_text)
        positive_words = set(opinion_lexicon.positive())
        negative_words = set(opinion_lexicon.negative())
        positive_word_count = len([word for word in processed_text.split() if word in positive_words])
        negative_word_count = len([word for word in processed_text.split() if word in negative_words])

        #debug
        print("input : ",input_data)
        print("sentence after cleaning : ",processed_review )
        print("sentence after processed text : ",processed_text)
        print("pos word count : ",positive_word_count)
        print("negative word count : ",negative_word_count )

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([processed_text])

        sequence = tokenizer.texts_to_sequences([processed_text])
        max_length=200

        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

        # Combine features (update if you include more features)
        additional_features = np.array([[positive_word_count,negative_word_count,description_length]])
        combined_features = np.hstack((padded_sequence, additional_features))
        return combined_features

    elif isinstance(input_data, str) and input_data.endswith('.csv'):
        # Handle file path
        df = pd.read_csv(input_data)
        df.dropna(subset=['Description'], inplace=True)
        df.Description = df.Description.apply(lambda x: remove_strings(x, strings_to_remove))
        df.Description = df.Description.apply(preprocess_text)
        df.Description = df.Description.apply(word_tokenize)
        df.Description = df.Description.apply(lemmatize_text)
        df.Description = df.Description.apply(lambda x: ' '.join(x))
        df['Description_Length'] = df['Description'].apply(len)
        df.Rating = df['Rating'].apply(categorize_rating)

        positive_words = set(opinion_lexicon.positive())
        negative_words = set(opinion_lexicon.negative())
        df['Positive_Word_Count'] = df['Description'].apply(lambda x: len([word for word in x.split() if word in positive_words]))
        df['Negative_Word_Count'] = df['Description'].apply(lambda x: len([word for word in x.split() if word in negative_words]))

        df.drop_duplicates(inplace=True)
        df = df[df['Rating'] > 0]
        df = df.dropna()
        tokenizer = Tokenizer() 
        tokenizer.fit_on_texts(df['Description'])
        sequences = tokenizer.texts_to_sequences(df['Description'])

        max_length = max(len(x) for x in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

        X_additional = df[['Positive_Word_Count', 'Negative_Word_Count', 'Description_Length']].values
        X_combined = np.hstack((padded_sequences, X_additional))
        return X_combined

    else:
        raise ValueError("Unsupported input type for preprocess_data")
    


def prepare_data(df):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['Description'])
    sequences = tokenizer.texts_to_sequences(df['Description'])

    max_length = max(len(x) for x in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    X_additional = df[['Positive_Word_Count', 'Negative_Word_Count', 'Description_Length']].values
    X_combined = np.hstack((padded_sequences, X_additional))
    y = df['Rating']

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = preprocess_data('data/combined_reviews.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)