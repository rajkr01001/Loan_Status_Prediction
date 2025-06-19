import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

st.title("Loan Status Predictor")

# Load and preprocess data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    df.drop("Loan_ID", axis=1, inplace=True)

    # Fill missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ['LoanAmount', 'Loan_Amount_Term']:
        df[col].fillna(df[col].median(), inplace=True)

    # Convert categorical to numeric
    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents', 'Loan_Status']:
        df[col] = le.fit_transform(df[col])

    return df

# Load and train model
def train_model(df):
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

# Predict function
def predict_loan_status(model, input_data):
    prediction = model.predict(input_data)
    return "Approved" if prediction[0] == 1 else "Not Approved"

# Load and train
with st.spinner("Training model..."):
    df = load_data()
    model = train_model(df)

# User input form
st.sidebar.header("Enter Applicant Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0.0)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0.0)
loan_amount_term = st.sidebar.selectbox("Loan Term (months)", [360.0, 120.0, 180.0, 240.0, 300.0, 60.0, 84.0, 12.0])
credit_history = st.sidebar.selectbox("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode inputs
input_dict = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 0 if education == "Graduate" else 1,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": ["Urban", "Semiurban", "Rural"].index(property_area),
}

input_df = pd.DataFrame([input_dict])

# Prediction button
if st.sidebar.button("Predict Loan Status"):
    result = predict_loan_status(model, input_df)
    st.success(f"Loan Application is likely to be: {result}")
