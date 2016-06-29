# Classifying and Predicting Review Helpfulness

## Overview
Machine Learning Model for classifying online written reviews as helpful vs. nonhelpful

## Ratings
Helpfulness reviews provided as X out of Y users rated as helpful.
Normalized with a Bayesian Prior of Median(X)/Median(Y)

## Features
TFIDF of Review Text 
Adjective/Adverb Usage
Comparative/Superlative Usage
Named Entities
Quantities
Length
*Text features processed with spaCy*

## Data
Amazon Electronics Review Dataset from http://jmcauley.ucsd.edu/data/amazon/
5-core Electronics Reviews
