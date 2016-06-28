from __future__ import unicode_literals 
import dill
import sqlite3

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib
from bs4 import BeautifulSoup

from spacy.en import English
from gensim.models import Word2Vec

from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords
import re

# Load Data
raw_df = pd.read_pickle('parsed_df.pkl')
df = raw_df.sample(n=5000, random_state=123456)
X_df = df.drop('help_rate', axis=1)
y_df = df['help_rate']

# Column Selection Transformer
class ColumnTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Select Feature Columns
    """
    def __init__(self, cols):
        self.cols = cols
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert(type(X) == pd.core.frame.DataFrame), 'Features must be a DataFrame'
        try:
            return X.loc[:,(self.cols)]
        except (TypeError, KeyError):
            raise TypeError('Column selection must be list of strings')

# Text Cleaning Transformer
class TextCleanTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    '''
    Cleans Raw Review Text for Processing
    '''
    def __init__(self, cols):
        self.cols = cols
        pass
    
    def cleaner(self, text):
        # Clean HTML and set to lowercase
        clean_ = BeautifulSoup(text, 'lxml').get_text().lower() 
        # Clear newlines
        clean_ = clean_.strip().replace("\n", " ").replace("\r", " ")
        return clean_
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        cleaned = X[self.cols[0]].apply(self.cleaner)
        return cleaned

# SpaCy Tokenizer
def spacy_tokenize(text):
    # Tokenize with spaCy
    parser = English(tagger=False, entity=False, parser=False, matcher=False)
    tokens = parser(text)
    
    #lemmatize
    lemmas = []
    for t in tokens:
        lemmas.append(t.lemma_.lower().strip() if t.lemma_ != '-PRON-' else t.lower_)
    tokens = lemmas
    
    # remove whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
        
    return tokens

# TFIDF Pipeline
tfidf_mod = Pipeline([
    ('select', ColumnTransformer(['text'])),
    ('clean', TextCleanTransformer(['text'])),
    ('vectorize', TfidfVectorizer(tokenizer=spacy_tokenize, ngram_range=(1,1))),
    ('enet', RidgeCV(cv=5))
    ])

# Fit Model
tfidf_mod.fit(X_df, y_df)

dill.dump(tfidf_mod, open('tfidf_mod', 'w'), recurse=True)