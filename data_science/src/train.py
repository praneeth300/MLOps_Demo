import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from huggingface_hub import HfApi
import joblib

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

# Start MLflow experiment
mlflow.set_experiment("churn_model_experiment")

# Try different models
models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=5, criterion='gini'),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

best_model = None
best_accuracy = 0.0
best_model_name = ""
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\nModel: {model_name}")
        print(f"Train Accuracy: {train_acc:.2f}")
        print(f"Test Accuracy: {test_acc:.2f}")
        print("\nTrain Classification Report:")
        print(classification_report(y_train, y_pred_train))
        print("Test Classification Report:")
        print(classification_report(y_test, y_pred_test))

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, y_pred_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

        if test_acc > best_accuracy:
            best_model = model
            best_accuracy = test_acc
            best_model_name = model_name
            joblib.dump(best_model, "best_churn_model.joblib")

print(f"\nBest model: {best_model_name} with Test Accuracy: {best_accuracy:.2f}")


api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id="praneeth232/test-model",
    repo_type="model",
)
