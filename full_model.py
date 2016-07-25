from __future__ import unicode_literals 
import dill
import sqlite3

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import re
from bs4 import BeautifulSoup
from spacy.en import English

import sklearn.metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

# Load Models
#MultiNBModel = dill.load(open('mnb_class'))     # acc = .591
#LogitModel = dill.load(open('logit_class'))     # acc = .677
ForestModel = dill.load(open('forest_class'))     # acc = .688
#SvmModel = dill.load(open('svm_class'))         # acc = 

#MultiNBTfidf = dill.load(open('mnb_tfidf'))     # acc = .653 (typext)
LogitTfidf = dill.load(open('logit_tfidf'))     # acc = .696 (text)
#ForestTfidf = dill.load(open('forest_tfidf'))   # acc = .642 (lemma)

#LogitW2V = dill.load(open('logit_w2v'))        # acc = .660 (lemma)
#NBW2V = dill.load(open('nb_w2v'))              # acc = .604 (lemma)

# Load Data
df = pd.read_pickle('parsed_df_wlem.pkl')

X_df = df.drop('help_class', axis=1)
y_df = df['help_class']

#Transformers
class PredTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Use predicted models as features
    """
    def __init__(self, pipe):
        self.pipe = pipe
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert(type(X) == pd.core.frame.DataFrame), 'Features must be a DataFrame'
        preds = self.pipe.predict(X)
        return preds

class ArrayTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Convert Feature Union to Array
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        assert(type(X) == np.ndarray), 'Features must be a numpy array'
        X_ar = X.reshape(2, len(X)/2).transpose()
        return X_ar	

# Feature Union
all_features = FeatureUnion([
    ('forest', PredTransformer(ForestModel)),
    ('tf_logit', PredTransformer(LogitTfidf))
    #,('w2v_logit', PredTransformer(LogitW2V))
    ])

# Full Pipeline CV
search = {"min_samples_split": [2, 10, 20],
          "max_depth": [None, 2, 5, 10],
          "min_samples_leaf": [1, 5, 10],
          "max_leaf_nodes": [None, 5, 10, 20]}
full_pipe = sk.pipeline.Pipeline([
    ('predictions', all_features),
    ('to_array', ArrayTransformer()),
    ('forest', GridSearchCV(RandomForestClassifier(), param_grid=search, cv=5, scoring='accuracy'))
    ])

full_pipe.fit(X_df, y_df)

print full_pipe.named_steps['forest'].best_params_
print full_pipe.named_steps['forest'].best_score_