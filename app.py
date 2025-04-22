from huggingface_hub import hf_hub_download
import joblib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Replace with your model repo
repo_id = "praneeth232/test-model"
filename = "best_churn_model.joblib"

# This fetches the file and gives you the local path
model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Telecom Customer Churn Prediction App")
st.write("This tool predicts customer churn risk based on their details. Enter the required information below.")
tenure = st.number_input("Tenure (Months with the company)", min_value=0, value=12)
Contract = st.selectbox("Type of Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=600.0)
InternetService = st.selectbox("Type of Internet Service", ["DSL", "Fiber optic", "No"])
TechSupport = st.selectbox("TechSupport", ["No", "No internet service", "Yes"])
OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "No internet service", "Yes"])
SeniorCitizen = st.selectbox("SeniorCitizen", ["Yes", "No"])
    

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'tenure': tenure,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'InternetService': InternetService,
    'TechSupport': TechSupport,
    'OnlineSecurity': OnlineSecurity,
    'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0
}])

# Apply Label Encoding
categorical_cols = ['Contract', 'PaperlessBilling', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity']

for col in categorical_cols:
    le = LabelEncoder()
    input_data[col] = le.fit_transform(input_data[col])
    
# Set classification threshold
classification_threshold = 0.5

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to **{result}**.")
