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

# Multinomial Naive Bayes Pipeline
features = ['length', 'dfine_pct', 'dcoarse_pct', 'ent_pct', 'quant_pct', 
            'sent_len', 'sent_fine', 'sent_coarse', 'sent_ent',  'sent_quant',
            #'score_pos', 'score_neg',
            'score_low', 'score_high']

mnb_mod = Pipeline([
                    ('select', ColumnTransformer(features)),
                    ('mnb', MultinomialNB())
                    ])

# Accuracy Score
acc = cross_val_score(mnb_mod, X_df, y_df, cv=5, scoring='accuracy').mean()
roc_auc = cross_val_score(mnb_mod, X_df, y_df, cv=5, scoring='roc_auc').mean()

# Fit Model
mnb_mod.fit(X_df, y_df)
dill.dump(mnb_mod, open('mnb_class', 'w'), recurse=True)