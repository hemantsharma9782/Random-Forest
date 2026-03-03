import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load Saved Model & Preprocessor
# -------------------------------
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessing.pkl")

st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer will purchase the new Wellness Package")

# -------------------------------
# User Input Section
# -------------------------------

st.header("Enter Customer Details")

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
AnnualIncome = st.number_input("Annual Income", min_value=0, value=50000)
FamilySize = st.number_input("Family Size", min_value=1, max_value=10, value=2)
NumberOfTrips = st.number_input("Number of Trips Last Year", min_value=0, value=2)

# Example categorical features
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Occupation = st.selectbox("Occupation", ["Salaried", "Business", "Student"])
PackageType = st.selectbox("Previous Package", 
                           ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict"):

    # Create dataframe from inputs
    input_data = pd.DataFrame({
        "Age": [Age],
        "AnnualIncome": [AnnualIncome],
        "FamilySize": [FamilySize],
        "NumberOfTrips": [NumberOfTrips],
        "MaritalStatus": [MaritalStatus],
        "Occupation": [Occupation],
        "PackageType": [PackageType]
    })

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]

    # Show result
    if prediction[0] == 1:
        st.success(f"Customer is likely to purchase (Probability: {probability:.2f})")
    else:
        st.error(f"Customer is NOT likely to purchase (Probability: {probability:.2f})")