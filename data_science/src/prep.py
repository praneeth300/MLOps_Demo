import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
DATASET_PATH = "hf://datasets/praneeth232/test/teleco_churn.csv"
TRAIN_OUTPUT_PATH = "data/train.csv"
TEST_OUTPUT_PATH = "data/test.csv"

# Read the dataset
try:
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading the dataset: {e}")
    raise  # Reraise exception to stop further execution

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

# Ensure output directories exist
os.makedirs(os.path.dirname(TRAIN_OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TEST_OUTPUT_PATH), exist_ok=True)

# Save train and test datasets
try:
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)
    print("Datasets saved successfully.")
except Exception as e:
    print(f"Error saving datasets: {e}")
