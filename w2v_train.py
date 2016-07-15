from __future__ import unicode_literals 
import dill

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

import re
from bs4 import BeautifulSoup
from spacy.en import English
from gensim.models import Word2Vec
from nltk.corpus import stopwords
	
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# Load Data
df = pd.read_pickle('parsed_df_wlem.pkl')

X_df = df[['lemma']]
#X_df = df[['text']]
y_df = df['help_class']

# List of Lemmas
def lemma_list(row):
	words = row.split()
	return [w for w in words if not w in stopwords.words("english")]   

# Average Word Vectors
def avg_vec(wordlist,size):
    '''
    Return an average of the embeddings vectors
    Return a vector of zero when all words not in training or not at min
    '''
    sumvec=np.zeros(shape=(1,size))
    wordcnt=0
    
    for w in wordlist:
        if w in model:
            sumvec += model[w]
            wordcnt +=1
    
    if wordcnt ==0:
        return sumvec
    
    else:
        return sumvec / wordcnt

# Produce lemma list for training
X_df['lem_list'] = X_df['lemma'].apply(lemma_list)
#X_df['lem_list'] = X_df['text'].apply(lemma_list)
X_train, X_test, y_train, y_test = train_test_split(X_df['lem_list'], y_df, test_size=0.2, random_state=123456)

#size of hidden layer (length of continuous word representation)
dimsize=400

# Train Word2Vec
model = Word2Vec(X_train.values, size=dimsize, window=5, min_count=5, workers=4)

#create average vector for train and test from model
#returned list of numpy arrays are then stacked 
X_train=np.concatenate([avg_vec(w,dimsize) for w in X_train])
X_test=np.concatenate([avg_vec(w,dimsize) for w in X_test])

# fit model
logit_w2v = LogisticRegression()
logit_w2v.fit(X_train, y_train)

dill.dump(logit_w2v, open('logit_w2v', 'w'), recurse=True)

y_pred=logit_w2v.predict(X_test)

# accuracy = .660, lemma
# accuracy = .654, word
print accuracy_score(y_test,y_pred)