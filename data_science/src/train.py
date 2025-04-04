import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


train_path = "hf://datasets/praneeth232/test/train.csv"
test_path = "hf://datasets/praneeth232/test/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = 'Churn'  # Assuming 'Churn' is the target column

y_train = train_df[target_col]
X_train = train_df.drop(columns=[target_col])
y_test = test_df[target_col]
X_test = test_df.drop(columns=[target_col])

model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
model.fit(X_train, y_train)


yhat_test = model.predict(X_test)
accuracy = accuracy_score(y_test, yhat_test)
print(f'Accuracy of Decision Tree classifier on test set: {accuracy:.2f}')

