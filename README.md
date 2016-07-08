# Classifying and Predicting Review Helpfulness

## Overview
Machine Learning Model for classifying online written reviews as helpful vs. nonhelpful

## Ratings
Helpfulness reviews provided as X out of Y users rated as helpful.

Normalized with a Bayesian Prior of Median(X)/Median(Y)

## Features
1. TFIDF of Lemmatized Review Text 
2. Adjective/Adverb Usage (per sentence and per review)
3. Comparative/Superlative Usage (per sentence and per review)
4. Named Entities (per sentence and per review)
5. Quantities (per sentence and per review)
6. Length
7. Score Rating above/below average

*Text features processed with spaCy*

## Files
* `parse_raw.py` pulls reviews, processes text, and generates features and lemmatized text
* `nb_class.py` trains a Multinomial Naive Bayes classification using constructed features. Model saved to `mnb_class` dill
* `logit_class.py` trains a Randomized Logistic Regression classification using constructed features. Model saved to `logit_class` dill
* `rf_class.py` trains a Random Forest classification using constructed features. Model saved to `forest_class` dill
* `logit_tfidf.py` trains a Logistic Regression classification using lemmatized TFIDF features. Model saved to `logit_tfidf` dill
* `rf_tfidf.py` trains a Random Forest classification using lemmatized TFIDF features. Model saved to `forest_tfidf` dill
* `full_model.py` aggregates models using bagging with random forest.
* `svm_tfidf.py` is unused. 

## Data
Amazon Electronics Review Dataset from http://jmcauley.ucsd.edu/data/amazon/

5-core Electronics Reviews

Currently trained on a balanced subsample of 75000 reviews of 1.6 million split between classes

