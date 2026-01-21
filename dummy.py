import streamlit as st
import pandas as pd
from joblib import load

# Load trained model
model = load('random_forest_model.joblib')

# App title
st.title("Customer Churn Prediction App")

st.header("Enter Customer Information")

# User inputs
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)

internet_service = st.selectbox(
    "Internet Service",
    ('DSL', 'Fiber optic', 'No')
)

contract = st.selectbox(
    "Contract",
    ('Month-to-month', 'One year', 'Two year')
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=0.0
)

# Manual label encoding (must match training)
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}

internet_service = label_mapping[internet_service]
contract = label_mapping[contract]

# 🔹 Predict Button
if st.button("Predict Churn"):
    prediction = model.predict([[
        tenure,
        internet_service,
        contract,
        monthly_charges,
        total_charges
    ]])

    st.header("Prediction Result")

    if prediction[0] == 0:
        st.success("✅ This customer is likely to stay.")
    else:
        st.error("⚠️ This customer is likely to churn.")
