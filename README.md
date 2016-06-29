# Classifying and Predicting Review Helpfulness

## Overview
Machine Learning Model for classifying online written reviews as helpful vs. nonhelpful

## Ratings
Helpfulness reviews provided as X out of Y users rated as helpful.

Normalized with a Bayesian Prior of Median(X)/Median(Y)

## Features
1. TFIDF of Review Text 
2. Adjective/Adverb Usage
3. Comparative/Superlative Usage
4. Named Entities
5. Quantities
6. Length

*Text features processed with spaCy*

## Data
Amazon Electronics Review Dataset from http://jmcauley.ucsd.edu/data/amazon/

5-core Electronics Reviews
