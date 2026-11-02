import pandas as pd
import numpy as np
import re
import emoji
import unicodedata
import nltk
from nltk.corpus import stopwords, opinion_lexicon, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

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

tokenizer = Tokenizer()
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

def process_review(review):
    review = preprocess_text(review)
    review = word_tokenize(review)
    review = lemmatize_text(review)
    
    review_seq = tokenizer.texts_to_sequences([review])
    sequences = pad_sequences(review_seq, maxlen=200, padding='post')
    
    pos_word_count = len([word for word in review if word in positive_words])
    neg_word_count = len([word for word in review if word in negative_words])
    desc_len = len(review)
    
    x_add = np.array([[pos_word_count, neg_word_count, desc_len]])
    x_comb = np.hstack((sequences, x_add))
    
    return x_comb
