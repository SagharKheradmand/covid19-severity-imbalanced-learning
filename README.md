
# COVID-19 Severity Prediction with Imbalanced Ensemble Learning

## Overview

This project investigates machine learning methods for predicting severe COVID-19 cases from clinical and demographic data under strong class imbalance.

The main approaches include:

- Hellinger Distance Decision Tree (HDDT)
- Bagging with Majority-Class Undersampling
- AdaBoost.M1 with SMOTE

The models are evaluated using imbalance-aware metrics such as minority-class recall, F1-score, AUC-ROC, and G-mean.

## Dataset

The dataset contains:

- 1,585 samples
- 40 input features
- 100 severe cases
- 1,485 non-severe cases
- Approximate imbalance ratio: 15:1

## Main Topics

- Imbalanced Classification
- Hellinger Distance Decision Trees
- Bagging
- Random Undersampling
- AdaBoost
- SMOTE
- Missing-Value Imputation
- Feature Selection
- Minority-Class Evaluation

## Technologies

Python · NumPy · Pandas · Matplotlib · scikit-learn · Jupyter Notebook

## Author

Saghar Kheradmand
