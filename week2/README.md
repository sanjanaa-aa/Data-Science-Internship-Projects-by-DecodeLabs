# Week 2: Credit Default Prediction using Machine Learning

## Project Overview
This project predicts whether a customer will default on their credit card payment in the next month using supervised machine learning techniques.

## Dataset
- Default of Credit Card Clients Dataset (UCI Repository)
- 30,000 customer records
- 24 input features
- 1 target variable

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)

## Project Workflow
1. Data Loading
2. Data Exploration
3. Missing Value Analysis
4. Feature Selection
5. Train-Test Split
6. SMOTE for Class Balancing
7. Logistic Regression
8. Random Forest Classification
9. Model Evaluation

## Evaluation Metrics
- Precision
- Recall
- ROC-AUC

## Results

### Logistic Regression
- Precision: 0.3865
- Recall: 0.5666
- ROC-AUC: 0.6573

### Random Forest
- Precision: 0.5067
- Recall: 0.4890
- ROC-AUC: 0.6778

## Conclusion
Random Forest achieved better overall performance compared to Logistic Regression and was selected as the final model.