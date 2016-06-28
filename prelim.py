from __future__ import unicode_literals 
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from spacy.en import English
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split


# Connect and Load Data into Dataframe
con = sqlite3.connect('~data/database.sqlite')

query = """
        SELECT Id as uid, ProductId as prodid,
               HelpfulnessNumerator as is_help, 
               HelpfulnessDenominator as num_help, 
               Score as score, Summary as summ, Text as text
        FROM Reviews 
        WHERE num_help >= 10
        """

df = pd.read_sql_query(query, con)
print(len(df))


# Create Bayes Ranking of Helpfulness
## Construct Additional Helpfulness counts
#df['not_help'] = df['num_help'] - df['is_help']
df['help_pct'] = df['is_help']/df['num_help']
## Set Priors (May need tweaking)
pr_help = df['is_help'].mean() - .5*df['is_help'].std()
pr_num = df['num_help'].mean() #+ df['num_help'].std()

## Generate Rating
df['help_rate'] = (df['is_help'] + pr_help)/(df['num_help'] + pr_num)

# Process Text
## Strip HTML 
df['text'] = df['text'].str.replace('<.*?>','')


# Train Models

# Load Check 
df.hist(column = 'help_rate')
plt.show()
print(df.head())

con.close()