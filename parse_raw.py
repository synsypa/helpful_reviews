from __future__ import unicode_literals 
import sqlite3
import gzip
import dill

import pandas as pd
import numpy as np

import re
from bs4 import BeautifulSoup
from spacy.en import English

# Read in Electronics Reviews from SQLite Database
con = sqlite3.connect('amazon_elec.sqlite')
query = """
        SELECT product_id as pid,
               is_help, num_help, 
               score, summary, text
        FROM reviews 
        WHERE num_help >= 5
        LIMIT 100000
        """
df = pd.read_sql_query(query, con)

### Text Cleaning
# Text Cleaning Function
def clean_text(text):
    """
    Return a list of cleaned text from the raw review
    """
    #Clean HTML and convert to lower case
    clean_ = BeautifulSoup(text, 'lxml').get_text().lower() 
    
    return clean_

df['text'] = df['text'].apply(clean_text)

### Generate Ratings
# Create counts of not_helpful, % helpful votes
df['not_help'] = df['num_help'] - df['is_help']
df['help_pct'] = df['is_help']/df['num_help']

# Create Priors for helpful, not helpful, total ratings
pr_help = df['is_help'].median()
pr_not = df['not_help'].median()
#pr_num = 5

# Create Weighted values
w_help = df['is_help'] + pr_help
w_not = df['not_help'] + pr_not
#w_num = df['num_help'] + pr_num

# Create bayesian ranking of helpfulness
df['help_rate'] = w_help/(w_help + w_not)
df['help_bin'] = pd.qcut(df['help_rate'], 10, range(1,11))
df['help_log'] = np.log(df['help_rate'] + 1) 

### Generate score average and deviation features
df['prod_score'] = df['score'].groupby(df['pid']).transform('mean')
df['prod_ct'] = df['score'].groupby(df['pid']).transform('count')
df['score_chg'] = df['score'] - df['prod_score']
df['score_diff'] = abs(df['score_chg'])

# Generate Positive/Negative Indicator
df['score_pos'] = np.where(df['score'] > 3, 1, 0)
df['score_neg'] = np.where(df['score'] < 3, 1, 0)
df['score_low'] = np.where(df['score_chg'] < 0, 1, 0)
df['score_high'] = np.where(df['score_chg'] > 0, 1, 0)

### Generate Text-based features
parser = English()#parser=False)#, entity=False, matcher=False)
df['parsed'] = df['text'].apply(parser)

# Function to count words in review
def count_wrd(spacy):
    words = 0
    for token in spacy:
        if token.pos_ not in ["PUNCT", "SYM", "X", "EOL", "SPACE"]:    
            words += 1
    return words 

# Count words per sentence
def count_sent(spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(count_wrd(sent))
    return 1. * sum(nsent)/len(nsent)

# Count # of adj/adv 
def coarse_desc(spacy):
    descw_ = 0
    for token in spacy:
        if token.pos_ in ["ADJ", "ADV"]:
            descw_ += 1
    return descw_

# Count adj/adv per sentence
def coarse_sent(spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(coarse_desc(sent))
    return 1. * sum(nsent)/len(nsent)

# Count # of comparatives/superlatives
def fine_desc(spacy):
    descw_ = 0
    for token in spacy:
        if token.tag_ in ["JJR", "JJS", "RBR", "RBS"]:
            descw_ += 1
    return descw_

# Count comp/super per sentence
def fine_sent(spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(fine_desc(sent))
    return 1. * sum(nsent)/len(nsent)

# Count # of named entities
def entity(spacy):
    entw_ = 0
    for token in spacy:
        if token.ent_type_ != "":
            entw_ += 1
    return entw_

# Count named entities per sentence
def ent_sent(spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(entity(sent))
    return 1. * sum(nsent)/len(nsent)

# Count # of Money or Quantity entities
def quant(spacy):
    entw_ = 0
    for token in spacy:
        if token.ent_type_ in ["QUANTITY", "MONEY", "PERCENT"]:
            entw_ += 1
    return entw_

# Count money/quant entities per sentence
def quant_sent(spacy):
    nsent = []
    for sent in spacy.sents:
        nsent.append(quant(sent))
    return 1. * sum(nsent)/len(nsent)


df['length'] = df['parsed'].apply(count_wrd)

df['desc_coarse'] = df['parsed'].apply(coarse_desc)
df['desc_fine'] = df['parsed'].apply(fine_desc)
df['dcoarse_pct'] = df['desc_coarse'] / df['length']
df['dcoarse_pct'] = df['dcoarse_pct'].fillna(0)
df['dfine_pct'] = df['desc_fine'] / df['length']
df['dfine_pct'] = df['dfine_pct'].fillna(0)
df['desc_ratio'] = df['desc_fine']/df['desc_coarse']
df['desc_ratio'] = df['desc_ratio'].fillna(0)

df['entities'] = df['parsed'].apply(entity)
df['quantities'] = df['parsed'].apply(quant)
df['ent_pct'] = df['entities'] / df['length']
df['ent_pct'] = df['ent_pct'].fillna(0)
df['quant_pct'] = df['quantities'] / df['length']
df['quant_pct'] = df['quant_pct'].fillna(0)

df['sent_len'] = df['parsed'].apply(count_sent)
df['sent_coarse'] = df['parsed'].apply(coarse_sent)
df['sent_fine'] = df['parsed'].apply(fine_sent)
df['sent_ent'] = df['parsed'].apply(ent_sent)
df['sent_quant'] = df['parsed'].apply(quant_sent)

df = df.drop('parsed', axis=1)

### Save to Pickle
df.to_pickle('parsed_df.pkl')