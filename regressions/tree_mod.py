from __future__ import unicode_literals 
import dill
import sqlite3

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from itertools import combinations

# Load Parsed Data
df = pd.read_pickle('parsed_df.pkl')

X_df = df.drop('help_rate', axis = 1)
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

# GridSearchCV Pipeline for Regression Tree
search = {"min_samples_split": [2, 10, 20],
          "max_depth": [None, 2, 5, 10],
          "min_samples_leaf": [1, 5, 10],
          "max_leaf_nodes": [None, 5, 10, 20]}
features = ['length', 'dfine_pct', 'dcoarse_pct', 'ent_pct', 'quant_pct', 
            'sent_len', 'sent_fine', 'sent_coarse', 'sent_ent',  'sent_quant']
tree_mod = Pipeline([
                    ('select', ColumnTransformer(features)),
                    ('rtree', GridSearchCV(RandomForestRegressor(), scoring='mean_squared_error', param_grid=search))
                    ])

# Fit Model
tree_mod.fit(X_df, y_df)

# Store Score
score = tree_mod.named_steps['rtree'].best_score_
rmse = (-1. * score)** .5

f_weights = tree_mod.named_steps['rtree'].best_estimator_.feature_importances_

# Save Model
dill.dump(tree_mod, open('basic_dtree', 'w'), recurse=True)