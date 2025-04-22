import streamlit as st
import pandas as pd
import joblib

# Load the trained model
def load_model():
    return joblib.load("best_churn_model.joblib")

model = load_model()

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


# Set classification threshold
classification_threshold = 0.5

# Predict button
# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to **{result}**.")
