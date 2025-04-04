import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/praneeth232/test/teleco_churn.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill missing values
print("Missing values handled.")

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])
print(f"Encoded categorical columns: {list(categorical_cols)}")

# Split dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Dataset split into train and test sets.")

train_df.to_csv("train.csv",index=False)
test_df.to_csv("test.csv",index=False)

api.upload_folder(
    folder_path="train.csv",
    repo_id="praneeth232/test",
    repo_type="dataset",
)

api.upload_folder(
    folder_path="test.csv",
    repo_id="praneeth232/test",
    repo_type="dataset",
)
