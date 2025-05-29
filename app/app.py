# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add model folder to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from utils import preprocess_input

# Load artifacts
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder_columns = joblib.load("model/encoder_columns.pkl")

st.title("📉 Customer Churn Prediction App")

# UI Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
cltv = st.number_input("CLTV", min_value=0)
services = st.slider("Total Services Opted", 0, 6)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

# Convert to DataFrame
input_dict = {
    'Gender': gender,
    'Senior Citizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'Tenure Months': tenure,
    'Phone Service': phone_service,
    'Paperless Billing': paperless_billing,
    'Monthly Charges': monthly_charges,
    'Total Charges': total_charges,
    'CLTV': cltv,
    'TotalServicesOpted': services,
    'Multiple Lines': multiple_lines,
    'Internet Service': internet_service,
    'Contract': contract,
    'Payment Method': payment_method
}
input_df = pd.DataFrame([input_dict])

# Label Encode binary fields (same as during training)
binary_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Paperless Billing']
for col in binary_cols:
    input_df[col] = 1 if input_df[col].iloc[0] == "Yes" or input_df[col].iloc[0] == "Male" else 0

# One-hot encode categorical fields
input_df = pd.get_dummies(input_df)

# Preprocess and predict
try:
    processed_input = preprocess_input(input_df, scaler, encoder_columns)

    if st.button("Predict Churn"):
        prediction = model.predict(processed_input)
        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is not likely to churn.")
except Exception as e:
    st.error(f"❌ Error during prediction: {e}")
