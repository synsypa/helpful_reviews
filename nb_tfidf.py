from __future__ import unicode_literals 
import dill

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import re
from bs4 import BeautifulSoup
from spacy.en import English
from nltk.corpus import stopwords

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

# Load data
df = pd.read_pickle('parsed_df_wlem.pkl')

X_df = df.drop('help_class', axis=1)
y_df = df['help_class']

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
        return X.loc[:,self.cols]

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

# Multinomial Naive Bayes
feature = 'text'
#feature = 'lemma'

nb_tfidf = Pipeline([
    ('select', ColumnTransformer(feature)),
    ('vectorize', TfidfVectorizer(ngram_range=(1,1), min_df=100, max_df=.95)),
    ('multi_nb', MultinomialNB()),
    ])

# Fit Model
nb_tfidf.fit(X_df, y_df)

dill.dump(nb_tfidf, open('mnb_tfidf', 'w'), recurse=True)

# Accuracy = .650 (lemma)
# Accuracy = .653 (text)
acc = cross_val_score(nb_tfidf, X_df, y_df, cv=5, scoring='accuracy').mean()
print acc