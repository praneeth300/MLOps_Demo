import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


api = HfApi()


train_path = "hf://datasets/praneeth232/test/train.csv"
test_path = "hf://datasets/praneeth232/test/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = 'Churn'  # Assuming 'Churn' is the target column

y_train = train_df[target_col]
X_train = train_df.drop(columns=[target_col])
y_test = test_df[target_col]
X_test = test_df.drop(columns=[target_col])

# Define model and hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

# Perform Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"\nTrain Accuracy: {accuracy_score(y_train, y_pred_train):.2f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")

print("\nTrain Classification Report:")
print(classification_report(y_train, y_pred_train))
print("Test Classification Report:")
print(classification_report(y_test, y_pred_test))

# Save best model
joblib.dump(best_model, "best_churn_model.joblib")


api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id="praneeth232/test-model",
    repo_type="model",
)
