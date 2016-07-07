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
df = pd.read_pickle('parsed_df.pkl')

# Construct Balanced Subset
#df['help_class'] = np.where(df['help_rate'] >= .8, 1, 0)

good_df = df[df['help_class'] == 1]
good_df = good_df.sample(n=20000, random_state=123456)

bad_df = df[df['help_class'] == 0]
bad_df = bad_df.sample(n=20000, random_state=123456)

cut_df = good_df.append(bad_df)

X_df = cut_df.drop('help_class', axis = 1)
y_df = cut_df['help_class']

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
    parser = English(entity=False, parser=False, matcher=False)
    tokens = parser(text)

    #lemmatize
    lemmas = []
    for t in tokens:
        lemmas.append(t.lemma_.lower().strip() if t.lemma_ != '-PRON-' else t.lower_)
    tokens = lemmas

    # remove stopwords
    tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
    
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

# Random Forest Pipeline
search = {'vectorize__min_df':[100, 200, 300],
            'vectorize__max_df':[.8, .9, .95]}
rf_tfidf = Pipeline([
    ('select', ColumnTransformer(['text'])),
    ('clean', TextCleanTransformer(['text'])),
    ('vectorize', TfidfVectorizer(tokenizer=spacy_tokenize, ngram_range=(1,1))),
    #('vectorize', TfidfVectorizer(ngram_range=(1,1), min_df=200, max_df=.95)),
    ('forest', RandomForestClassifier(max_leaf_nodes=None, max_depth=None, min_samples_leaf=10, min_samples_split=200))
    ])

# Fit Model
grid = GridSearchCV(rf_tfidf, param_grid=search, cv=5, scoring='accuracy')
grid.fit(X_df, y_df)
#rf_tfidf.fit(X_df, y_df)
print grid.best_params_

#dill.dump(rf_tfidf, open('forest_tfidf', 'w'), recurse=True)
dill.dump(grid, open('forest_tfidf', 'w'), recurse=True)

# Accuracy
acc = grid.best_score_
print acc