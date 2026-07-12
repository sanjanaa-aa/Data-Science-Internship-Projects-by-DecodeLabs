import pandas as pd

df = pd.read_excel(
    "Week2/default of credit card clients.xls",
    header=1
)

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nClass Distribution:")
print(df["default payment next month"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df.drop("default payment next month", axis=1)
y = df["default payment next month"]

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

from sklearn.model_selection import train_test_split

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set Shape:")
print(X_train.shape)

print("\nTesting Set Shape:")
print(X_test.shape)

from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

from sklearn.linear_model import LogisticRegression

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_smote, y_train_smote)

print("\nLogistic Regression Model Trained Successfully!")

# Make Logistic Regression Predictions
y_pred_lr = lr_model.predict(X_test)

print("\nPredictions Generated Successfully!")

from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Logistic Regression Evaluation
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
roc_auc = roc_auc_score(y_test, y_pred_lr)

print("\nLogistic Regression Results")
print("Precision:", precision)
print("Recall:", recall)
print("ROC-AUC:", roc_auc)

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_smote, y_train_smote)

print("\nRandom Forest Model Trained Successfully!")

# Random Forest Predictions
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Predictions Generated!")

# Random Forest Evaluation
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_pred_rf)

print("\nRandom Forest Results")
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("ROC-AUC:", rf_roc_auc)
print("\n========== FINAL COMPARISON ==========")
print(f"Logistic Regression ROC-AUC: {roc_auc:.4f}")
print(f"Random Forest ROC-AUC: {rf_roc_auc:.4f}")

if rf_roc_auc > roc_auc:
    print("\nBest Model: Random Forest")
else:
    print("\nBest Model: Logistic Regression")
