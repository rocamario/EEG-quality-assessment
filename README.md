# EEG Signal Quality Assessment - Kaggle Challenge

This repository contains the code and experiments performed for the Kaggle competition [EEG Signal Quality Analysis by Beacon Biosignals](https://www.kaggle.com/competitions/eeg-signal-quality-analysis-by-beacon-biosignals/overview).

## Overview

The goal of the challenge is to classify EEG signals into "good" and "bad" categories based on their quality. The code provided here implements several models and techniques, including feature engineering, model selection, and hyperparameter tuning, to achieve the best classification performance.

## Features Engineered

We extracted features from EEG signals across three domains:

### Time-Domain Features:
- Statistical measures: mean, variance, skewness, kurtosis.
- Hjorth parameters: activity, mobility, and complexity.
- Envelope statistics: mean, max, min, standard deviation.

### Frequency-Domain Features:
- Power spectral density in various EEG bands: delta, theta, alpha, beta, gamma.
- Relative power in each frequency band.
- Energy in predefined frequency bands.

### Complexity Measures:
- Entropy measures: Shannon entropy, spectral entropy, sample entropy.
- Wavelet packet decomposition for time-frequency representations.

## Here, we exploit two rather different paths. The first one is searching for the best set of extracted features, and the second one relates to finding the optimal classifier using AutoML package. 

## 1) Work on the efficient extracted features

By optimizing the XGBoost classifier using Grid-Search, this approach has been implemented using two settings of 1) preloaded extracted features or 2) extracting features from scratch in "Methods_XGBOOST_features_exploring_pre_extracted features.ipynb" and "Methods_XGBOOST_features_exploring_full.ipynb", respectively. For using the former, one needs to have the extratcted features file which have been sent by email to professor.

## 2) Work on selecting the best classifier pipeline Model

Several machine learning models were explored and evaluated:
- **Logistic Regression**: As a baseline model.
- **Random Forests**: For interpretability.
- **XGBoost**: For handling feature interactions.
- **K-Nearest Neighbors (KNN)**: Yielding the best performance (0.75 Cohen's Kappa).
- **LightGBM (LGBM)**: For fast training and handling large datasets.
- **Lazy Classifier**: For quick benchmarking.
- **AutoML**: Used for automated model optimization, achieving a Cohen's Kappa score of 0.76.

## Results

The best performance achieved was a **Cohen's Kappa score of 0.76** using AutoML. KNN with automated feature selection performed well with a Cohen's Kappa score of 0.75.
