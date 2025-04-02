import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Read dataset
df = pd.read_csv("hf://datasets/praneeth232/test/teleco_churn.csv")

# Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Split dataset into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# Save train and test datasets
train_df.to_csv("data", "train.csv"), index=False)
test_df.to_csv("data", "test.csv"), index=False)


