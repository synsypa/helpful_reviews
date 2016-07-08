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
MultiNBModel = dill.load(open('mnb_class'))
LogitModel = dill.load(open('logit_class'))
ForestModel = dill.load(open('forest_class'))
#LogitTfidf = dill.load(open('logit_tfidf'))
ForestTfidf = dill.load(open('forest_tfidf'))

# Text Feature Transformer
class LangTransformer(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    Transform incoming data for prediction
    """
    def __init__(self):
        self.parser = English()
        pass
    
    # Function to lemmatize words in review
    def lemma_text(self, spacy):
        return ' '.join(token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_)
    
    # Function to count words in review
    def count_wrd(self, spacy):
    words = 0
    for token in spacy:
        if token.pos_ not in ["PUNCT", "SYM", "X", "EOL", "SPACE"]:    
            words += 1
    return words 

    # Count words per sentence
    def count_sent(self, spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(count_wrd(sent))
    return 1. * sum(nsent)/len(nsent)

    # Count # of adj/adv 
    def coarse_desc(self, spacy):
    descw_ = 0
    for token in spacy:
        if token.pos_ in ["ADJ", "ADV"]:
            descw_ += 1
    return descw_

    # Count adj/adv per sentence
    def coarse_sent(self, spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(coarse_desc(sent))
    return 1. * sum(nsent)/len(nsent)

    # Count # of comparatives/superlatives
    def fine_desc(self, spacy):
    descw_ = 0
    for token in spacy:
        if token.tag_ in ["JJR", "JJS", "RBR", "RBS"]:
            descw_ += 1
    return descw_

    # Count comp/super per sentence
    def fine_sent(self, spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(fine_desc(sent))
    return 1. * sum(nsent)/len(nsent)

    # Count # of named entities
    def entity(self, spacy):
    entw_ = 0
    for token in spacy:
        if token.ent_type_ != "":
            entw_ += 1
    return entw_

    # Count named entities per sentence
    def ent_sent(self, spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(entity(sent))
    return 1. * sum(nsent)/len(nsent)

    # Count # of Money or Quantity entities
    def quant(self, spacy):
    entw_ = 0
    for token in spacy:
        if token.ent_type_ in ["QUANTITY", "MONEY", "PERCENT"]:
            entw_ += 1
    return entw_

    # Count money/quant entities per sentence
    def quant_sent(self, spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(quant(sent))
    return 1. * sum(nsent)/len(nsent)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        assert(type(X) == pd.core.frame.DataFrame), 'Input must be a DataFrame'
        X['parsed_'] = X['text'].apply(self.parser)
        
        X['lemma'] = X['parsed_'].apply(self.lemma_text)
        X['length'] = X['parsed_'].apply(self.count_wrd)
        
        X['desc_coarse'] = X['parsed_'].apply(self.coarse_desc)
        X['desc_fine'] = X['parsed_'].apply(self.fine_desc)
        X['dcoarse_pct'] = X['desc_coarse'] / X['length']
        X['dcoarse_pct'] = X['dcoarse_pct'].fillna(0)
        X['dfine_pct'] = X['desc_fine'] / X['length']
        X['dfine_pct'] = X['dfine_pct'].fillna(0)
        X['desc_ratio'] = X['desc_fine'] / X['desc_coarse']
        X['desc_ratio'] = X['desc_ratio'].fillna(0)

        X['entities'] = X['parsed_'].apply(self.entity)
        X['quantities'] = X['parsed_'].apply(self.quant)
        X['ent_pct'] = X['entities'] / X['length']
        X['ent_pct'] = X['ent_pct'].fillna(0)
        X['quant_pct'] = X['quantities'] / X['length']
        X['quant_pct'] = X['quant_pct'].fillna(0)

        X['sent_len'] = X['parsed_'].apply(self.count_sent)
        X['sent_coarse'] = X['parsed_'].apply(self.coarse_sent)
        X['sent_fine'] = X['parsed_'].apply(self.fine_sent)
        X['sent_ent'] = X['parsed_'].apply(self.ent_sent)
        X['sent_quant'] = X['parsed_'].apply(self.quant_sent)
        
        return X.drop('parsed_', axis=1)

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
        X_ar = X.reshape(5, len(X)/5).transpose()
        return X_ar

# Feature Union
all_features = FeatureUnion([
    ('multiNB', PredTransformer(MultiNBModel)),
    ('logit', PredTransformer(LogitModel)),
    ('forest', PredTransformer(ForestModel)),
    ('tf_logit', PredTransformer(LogitTfidf)),
    ('tf_forest', PredTransformer(ForestTfidf))
    ])

# Full Pipelinee
search = {"forest__min_samples_split": [2, 10, 20],
          "forest__max_depth": [None, 2, 5, 10],
          "forest__min_samples_leaf": [1, 5, 10],
          "forest__max_leaf_nodes": [None, 5, 10, 20]}
full_pipe = sk.pipeline.Pipeline([
    ('predictions', all_features),
    ('to_array', ArrayTransformer()),
    ('forest', RandomForestClassifier())
    ])

full_grid = GridSearchCV(full_pipe, param_grid=search, cv=3, scoring='accuracy')
full_grid.fit(X_df, y_df)

print full_grid.best_params_
print full_grid.best_score_