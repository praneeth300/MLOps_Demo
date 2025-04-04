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

model = DecisionTreeClassifier(criterion="gini", max_depth=5)
model.fit(X_train, y_train)


yhat_test = model.predict(X_test)
accuracy = accuracy_score(y_test, yhat_test)
print(f'Accuracy of Decision Tree classifier on test set: {accuracy:.2f}')

joblib.dump(model,"churn_model.joblib")


api.upload_file(
    path_or_fileobj="churn_model.joblib",
    path_in_repo="churn_model.joblib",
    repo_id="praneeth232/test-model",
    repo_type="model",
)
