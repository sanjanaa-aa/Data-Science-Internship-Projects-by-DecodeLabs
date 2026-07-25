# Week 4: Sentiment Analysis using Natural Language Processing (NLP)

## Project Overview

This project performs Sentiment Analysis on customer reviews collected from Amazon, IMDb, and Yelp datasets. The objective is to classify reviews as Positive or Negative using Natural Language Processing (NLP) techniques and Machine Learning.

## Dataset

The dataset contains labeled reviews from three sources:

- Amazon Reviews
- IMDb Movie Reviews
- Yelp Restaurant Reviews

### Dataset Statistics

- Total Reviews: 2748
- Features:
  - Review
  - Sentiment
- Positive Reviews: 1386
- Negative Reviews: 1362

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-Learn

## Project Workflow

### 1. Data Loading
- Loaded Amazon, IMDb, and Yelp review datasets.
- Combined all datasets into a single DataFrame.

### 2. Data Exploration
- Checked dataset shape.
- Verified column names.
- Analyzed class distribution.
- Checked missing values.

### 3. Text Preprocessing
- Tokenization
- Stop-word Removal
- Lemmatization

### 4. Feature Extraction
- TF-IDF Vectorization
- Converted text reviews into numerical features.

### 5. Machine Learning
- Train-Test Split (80:20)
- Logistic Regression Classifier

### 6. Model Evaluation
- Accuracy Score
- Precision
- Recall
- F1-Score
- Classification Report

## Results

### Logistic Regression Performance

- Accuracy: 82%

#### Classification Report

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Negative (0) | 0.83 | 0.83 | 0.83 |
| Positive (1) | 0.81 | 0.81 | 0.81 |

## Files

- sentiment_analysis.py
- README.md
- requirements.txt
- results.txt

## Conclusion

The sentiment analysis model successfully classified customer reviews with an accuracy of 82%. Text preprocessing and TF-IDF vectorization significantly improved the model's ability to understand review sentiment.