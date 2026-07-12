# Week 2: Credit Default Prediction using Machine Learning

## Description

This project predicts whether a customer is likely to default on their credit card payment in the next month using machine learning techniques. The project includes data preprocessing, class balancing using SMOTE, model training, and performance evaluation.

## Dataset

- Dataset: Default of Credit Card Clients Dataset (UCI Machine Learning Repository)
- Total Records: 30,000
- Total Features: 24
- Target Variable: Default Payment Next Month

## Objectives

- Analyze customer credit data.
- Handle class imbalance using SMOTE.
- Train machine learning models.
- Compare model performance.
- Predict credit card default risk.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- VS Code

## Project Workflow

### 1. Data Loading
- Loaded the credit card clients dataset.
- Explored dataset structure and dimensions.

### 2. Data Exploration
- Checked column names and dataset shape.
- Analyzed class distribution.

### 3. Missing Value Analysis
- Verified that the dataset contains no missing values.

### 4. Feature Selection
- Selected input features (X).
- Selected target variable (y).

### 5. Train-Test Split
- Split data into training and testing sets using an 80:20 ratio.

### 6. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Oversampling Technique).
- Balanced default and non-default classes.

### 7. Logistic Regression Model
- Trained Logistic Regression classifier.
- Generated predictions on test data.

### 8. Random Forest Model
- Trained Random Forest classifier.
- Generated predictions on test data.

### 9. Model Evaluation
Evaluated models using:
- Precision
- Recall
- ROC-AUC Score

## Results

### Logistic Regression

- Precision: 0.3865
- Recall: 0.5666
- ROC-AUC: 0.6573

### Random Forest

- Precision: 0.5067
- Recall: 0.4890
- ROC-AUC: 0.6778

## Final Comparison

| Model | Precision | Recall | ROC-AUC |
|---------|---------|---------|---------|
| Logistic Regression | 0.3865 | 0.5666 | 0.6573 |
| Random Forest | 0.5067 | 0.4890 | 0.6778 |

## Conclusion

- Both models successfully predicted credit default risk.
- Random Forest achieved the highest ROC-AUC score.
- Random Forest performed better overall and was selected as the final model.

## Project Files

- `credit_default_prediction.py`
- `default of credit card clients.xls`
- `results.txt`
- `README.md`

## How to Run

```bash
python credit_default_prediction.py
```

## Output

- Data loaded and preprocessed successfully.
- Class imbalance handled using SMOTE.
- Logistic Regression and Random Forest models trained.
- Performance metrics generated.
- Final model comparison completed.