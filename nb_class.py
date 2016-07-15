from __future__ import unicode_literals 
import dill

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.cross_validation import cross_val_score

# Read Data
df = pd.read_pickle('parsed_df_wlem.pkl')

X_df = df.drop(['help_class', 'text'], axis = 1)
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

# Multinomial Naive Bayes Pipeline
features = ['length', 'dfine_pct', 'dcoarse_pct', 'ent_pct', 'quant_pct', 
            'sent_len', 'sent_fine', 'sent_coarse', 'sent_ent',  'sent_quant',
            #'score_pos', 'score_neg',
            'score_low', 'score_high']

mnb_mod = Pipeline([
                    ('select', ColumnTransformer(features)),
                    ('mnb', MultinomialNB())
                    ])

# Accuracy Score = .591
acc = cross_val_score(mnb_mod, X_df, y_df, cv=5, scoring='accuracy').mean()
print acc
#roc_auc = cross_val_score(mnb_mod, X_df, y_df, cv=5, scoring='roc_auc').mean()

# Fit Model
mnb_mod.fit(X_df, y_df)
dill.dump(mnb_mod, open('mnb_class', 'w'), recurse=True)