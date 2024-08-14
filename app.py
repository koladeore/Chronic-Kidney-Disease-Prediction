import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load the trained model and dataset for column reference
df = pd.read_csv('heart.csv')

# Check if the model file exists
if os.path.exists('ckd_model.pkl'):
    model = joblib.load('ckd_model.pkl')
    st.write("Model loaded successfully!")
else:
    st.error("Model file not found. Please train the model first.")
    st.stop()  # Stop the script if the model is not found

# Create a Streamlit app
st.title('Chronic Kidney Disease Prediction')

# Mapping of columns to full names
column_full_names = {
    "Bp": "Blood Pressure",
    "Sg": "Specific Gravity",
    "Al": "Albumin",
    "Su": "Sugar",
    "Rbc": "Red Blood Cell",
    "Bu": "Blood Urea",
    "Sc": "Serum Creatinine",
    "Sod": "Sodium",
    "Pot": "Potassium",
    "Hemo": "Hemoglobin",
    "Wbcc": "White Blood Cell Count",
    "Rbcc": "Red Blood Cell Count",
    "Htn": "Hypertension",
}

# Input fields using full names
inputs = []
for col in df.drop(columns=['Class']).columns:
    full_name = column_full_names.get(col, col)  # Get full name or default to column name
    val = st.number_input(f'Enter {full_name}', value=float(df[col].mean()))
    inputs.append(val)

# Convert the inputs to a DataFrame for prediction
inputs = np.array(inputs).reshape(1, -1)
prediction = model.predict(inputs)

# Display the prediction
predicted_class = 'Yes' if prediction[0] >= 0.5 else 'No'
st.write(f'Prediction: {predicted_class}')
