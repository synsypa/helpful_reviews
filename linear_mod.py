from __future__ import unicode_literals 
import dill
import sqlite3

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from sklearn.cross_validation import KFold, train_test_split, cross_val_score
from itertools import combinations

# Load Data
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

# 5 Fold Cross-Validation for Linear Regression
#features = ['length', 'dfine_pct', 'dcoarse_pct', 'desc_ratio', 'ent_pct', 'quant_pct']
#for f in range(len(features)):  
#    f_try = combinations(features, f+1)
#    for group in f_try:
#        ins = list(group)
#        basic_mod = Pipeline([
#                            ('select', ColumnTransformer(ins)),
#                            ('linreg', LinearRegression(normalize=True))
#                            ])
#       mse = cross_val_score(basic_mod, X_df, y_df, cv=5, scoring='mean_squared_error')
#        r2 = cross_val_score(basic_mod, X_df, y_df, cv=5, scoring='r2').mean()
#        rmse = (-1.*mse.mean()) ** (.5)
#        print ins
#        print rmse
#        print r2
#        print

# Pipeline for cross-validated linear regression
lin_mod = Pipeline([
    ('select', ColumnTransformer([u'length', u'ent_pct', u'quant_pct', u'score_diff'])),
    ('linreg', LinearRegression(normalize=True))
    ])

# Fit model
lin_mod.fit(X_df, y_df)

# Store Scores
scores = cross_val_score(lin_mod, X_df, y_df, cv=5, scoring='mean_squared_error')
rmse = (-1.*scores.mean()) ** (.5)

f_weights = cvd_mod.named_steps['linreg'].coef_

dill.dump(lin_mod, open('basic_linreg', 'w'), recurse=True)