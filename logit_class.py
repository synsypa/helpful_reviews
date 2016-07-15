from __future__ import unicode_literals 
import dill

import pandas as pd
import numpy as np
import sklearn as sk

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression

from sklearn.cross_validation import cross_val_score

# sklearn.linear_model.RandomizedLogisticRegression uses a depreciated function
# ignore the depreciation error
import warnings
warnings.filterwarnings('ignore')

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

# Pipeline for Randomized Logit Regression selected Logit
features = ['length', 'dfine_pct', 'dcoarse_pct', 'ent_pct', 'quant_pct', 
            'sent_len', 'sent_fine', 'sent_coarse', 'sent_ent',  'sent_quant',
            #'score_pos', 'score_neg',
            'score_low', 'score_high']

rlr_mod = Pipeline([
                    ('select', ColumnTransformer(features)),
                    ('rlr', RandomizedLogisticRegression(random_state=123456)),
                    ('logit', LogisticRegression())
                    ])

# Accuracy Score = .677
acc = cross_val_score(rlr_mod, X_df, y_df, cv=5, scoring='accuracy').mean()
print acc


# Fit Model
rlr_mod.fit(X_df, y_df)
dill.dump(rlr_mod, open('logit_class', 'w'), recurse=True)

# Coefficient Weights 
weights = dict(zip(features, rlr_mod.named_steps['rlr'].scores_))
#dill.dump(weights, open('logit_coef', 'w'), recurse=True)

