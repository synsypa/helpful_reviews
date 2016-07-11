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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

# Load data
#df = pd.read_pickle('parsed_df.pkl')
df = pd.read_pickle('parsed_df_wlem.pkl')

#X_df = df.drop('help_class', axis = 1)
X_df = df['lemma']
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

# Random Forest Pipeline
## min leaf 5 /  max leaf = None / max depth = 50 / min split = 250: acc = .640
## min leaf 5 /  max leaf = None / max depth = 300 / min split = 250: acc = .643
## min leaf 20 /  max leaf = None / max depth = 300 / min split = 250: acc = .639

rf_tfidf = Pipeline([
    ('vectorize', TfidfVectorizer(ngram_range=(1,1), min_df=100, max_df=.95)),
    ('forest', RandomForestClassifier(max_leaf_nodes=None, max_depth=300, min_samples_split=250, min_samples_leaf=5)),
    ])

# Fit Model
rf_tfidf.fit(X_df, y_df)

#grid = GridSearchCV(rf_tfidf, param_grid=search, cv=5, scoring='accuracy')
#grid.fit(X_df, y_df)
#print grid.best_params_

dill.dump(rf_tfidf, open('forest_tfidf', 'w'), recurse=True)
#dill.dump(grid, open('forest_tfidf', 'w'), recurse=True)

# Accuracy = .642
acc = cross_val_score(rf_tfidf, X_df, y_df, cv=5, scoring='accuracy').mean()
#acc = grid.best_score_
print acc