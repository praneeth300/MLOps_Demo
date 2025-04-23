from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder


# Initialize Flask apps
app = Flask("Telecom Customer Churn Predictor")


# Replace with your model repoa
repo_id = "praneeth232/test-model"
filename = "best_churn_model.joblib"


# This fetches the file and gives you the local path
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the model
model = joblib.load(model_path)

# Define categorical columns to encode
categorical_cols = [
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'InternetService', 'TechSupport', 'OnlineSecurity'
]

# Utility function to encode input using fresh LabelEncoders
def encode_input(input_df):
    for col in categorical_cols:
        le = LabelEncoder()
        input_df[col] = le.fit_transform(input_df[col])
    return input_df

@app.get('/')
def home():
    return "Welcome to the Telecom Customer Churn Prediction API!"

@app.post('/v1/customer')
def predict_churn():
    customer_data = request.get_json()

    sample = {
        'tenure': customer_data['tenure'],
        'Contract': customer_data['Contract'],
        'PaperlessBilling': customer_data['PaperlessBilling'],
        'PaymentMethod': customer_data['PaymentMethod'],
        'MonthlyCharges': customer_data['MonthlyCharges'],
        'TotalCharges': customer_data['TotalCharges'],
        'InternetService': customer_data['InternetService'],
        'TechSupport': customer_data['TechSupport'],
        'OnlineSecurity': customer_data['OnlineSecurity'],
        'SeniorCitizen': 1 if customer_data['SeniorCitizen'] == "Yes" else 0
    }

    input_df = pd.DataFrame([sample])
    input_df = encode_input(input_df)

    prediction = model.predict(input_df).tolist()[0]
    label = "churn" if prediction == 1 else "not churn"

    return jsonify({'Churn expected?': label})

@app.post('/v1/customerbatch')
def predict_churn_batch():
    file = request.files['file']
    input_df = pd.read_csv(file)

    # Convert SeniorCitizen to numeric
    input_df['SeniorCitizen'] = input_df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    input_df = encode_input(input_df)
    predictions = model.predict(input_df).tolist()
    labels = ['churn' if x == 1 else 'not churn' for x in predictions]

    return jsonify({'predictions': labels})

if __name__ == '__main__':
    app.run(debug=True)
