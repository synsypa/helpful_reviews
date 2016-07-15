# Classifying and Predicting Review Helpfulness

## Overview
Machine Learning Model for classifying online written reviews as helpful vs. nonhelpful

## Ratings
Helpfulness reviews provided as X out of Y users rated as helpful.

Normalized with a Bayesian Prior of Median(X)/Median(Y)

## Features
1. TFIDF of Review Text 
2. Word2Vec of Lemmatized Review Text
2. Adjective/Adverb Usage (per sentence and per review)
3. Comparative/Superlative Usage (per sentence and per review)
4. Named Entities (per sentence and per review)
5. Quantities (per sentence and per review)
6. Length (sentence and review)
7. Score Rating above/below average

*Text features processed with spaCy*

## Files
* `parse_raw.py` pulls reviews, processes text, and generates features and lemmatized text. Saves resultant dataframe to `parsed_df_wlem.plk`.


* `nb_class.py` trains a Multinomial Naive Bayes classification using constructed features. Model saved to `mnb_class` dill (Accuracy = **59.1%**)
* `logit_class.py` trains a Randomized Logistic Regression classification using constructed features. Model saved to `logit_class` dill (Accuracy = **67.7%**)
* `rf_class.py` trains a Random Forest classification using constructed features. Model saved to `forest_class` dill (Accuracy = **68.6%**)
* `svm_class.py` trains a Support Vector Machine classification using constructed features (Kernal is determined via GridSearchCV). Model saved to `svm_class` dill (Accuracy = **%**)


* `mnb_tfidf.py` trains a Multinomial Naive Bayes classification using TFIDF features. (Includes code for using lemmatized text as features). Model saved to `mnb_tfidf` dill (Accuracy = **65.3%** (text) **65.0%** (lemma))
* `logit_tfidf.py` trains a Logistic Regression classification using TFIDF features. (Includes code for using lemmatized text as features). Model saved to `logit_tfidf` dill (Accuracy = **69.6%** (text) **69.5%** (lemma))
* `rf_tfidf.py` trains a Random Forest classification using TFIDF features. (Includes code for using lemmatized text as features). Model saved to `forest_tfidf` dill (Accuracy = **%** (text) **64.2%** (lemma))

* `nb_w2v.py` trains a Gaussian Naive Bayes classification using word vectors generated from a word2vec model based on lemmatized text. (Includes code for using non-lemma text as features). Model saved to `nb_w2v` dill (Accuracy = **%** (text) **%** (lemma))
* `logit_w2v.py` trains a Logistic Regression classification using word vectors generated from a word2vec model based on lemmatized text. (Includes code for using non-lemma text as features). Model saved to `logit_w2v` dill (Accuracy = **65.4%** (text) **66.0%** (lemma))

* `full_model.py` aggregates models using bagging with random forest.

## Data
Amazon Electronics Review Dataset from http://jmcauley.ucsd.edu/data/amazon/

5-core Electronics Reviews

Currently trained on a balanced subsample of 75000 reviews of 1.6 million split between classes

