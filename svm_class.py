from __future__ import unicode_literals 
import dill

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from itertools import combinations

# Load Data
df = pd.read_pickle('parsed_df_wlem.pkl')

X_df = df.drop(['help_class', 'text', 'lemma'], axis = 1)
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

# GridSearchCV SVM Pipeline
features = ['length', 'dfine_pct', 'dcoarse_pct', 'ent_pct', 'quant_pct', 
            'sent_len', 'sent_fine', 'sent_coarse', 'sent_ent',  'sent_quant',
            'score_low', 'score_high']

search = {'kernel': [str('linear'), str('poly'), str('rbf')]}

svm_mod = Pipeline([
    ('select', ColumnTransformer(features)),
    ('svm', GridSearchCV(SVC(), param_grid=search, cv=5, scoring='accuracy'))
    ])
svm_mod.fit(X_df, y_df)

# Accuracy
acc = rf_mod.named_steps['svm'].best_score_
print acc 

# Fit Model
dill.dump(svm_mod, open('svm_class', 'w'), recurse=True)