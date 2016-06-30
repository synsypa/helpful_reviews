from __future__ import unicode_literals 
import dill

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

# Load Parsed Data
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

# GridSearchCV Pipeline for Regression Tree
features = ['length', 'dfine_pct', 'dcoarse_pct', 'ent_pct', 'quant_pct', 
            'sent_len', 'sent_fine', 'sent_coarse', 'sent_ent',  'sent_quant',
            #'score_pos', 'score_neg',
            'score_low', 'score_high']
search = {"min_samples_split": [2, 10, 20],
          "max_depth": [None, 2, 5, 10],
          "min_samples_leaf": [1, 5, 10],
          "max_leaf_nodes": [None, 5, 10, 20]}

rf_mod = Pipeline([
                    ('select', ColumnTransformer(features)),
                    ('forest', GridSearchCV(RandomForestClassifier(), param_grid=search, cv=5, scoring='accuracy'))
                    ])

# Fit Model
rf_mod.fit(X_df, y_df)
dill.dump(rf_mod, open('forest_class', 'w'), recurse=True)

# Store Score
acc = rf_mod.named_steps['forest'].best_score_
weights = dict(zip(features, rf_mod.named_steps['forest'].best_estimator_.feature_importances_))