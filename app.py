import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image

# Load the dataset for column reference
df = pd.read_csv('heart.csv')

# Check if the model file exists
if os.path.exists('ckd_model.pkl'):
    model = joblib.load('ckd_model.pkl')
    st.write("Model loaded successfully!")
else:
    st.error("Model file not found. Please train the model first.")
    st.stop()  # Stop the script if the model is not found

# Load metrics if available
if os.path.exists('metrics.pkl'):
    metrics = joblib.load('metrics.pkl')
else:
    st.error("Metrics file not found. Please ensure the model is trained and metrics are saved.")
    st.stop()  # Stop the script if metrics are not found

# Create a Streamlit app
st.title('Chronic Kidney Disease Prediction using Multiple Linear Regression')

# Mapping of columns to full names for better UX
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

# Display additional metrics and plots
st.header('Model Performance Metrics')
st.write(f'Precision: {metrics["precision"]}')
st.write(f'F1 Score: {metrics["f1_score"]}')
st.write(f'Accuracy: {metrics["accuracy"]}')
st.write(f'Mean Squared Error: {metrics["mse"]}')

# Display Confusion Matrix
if os.path.exists('confusion_matrix.png'):
    st.image(Image.open('confusion_matrix.png'), caption='Confusion Matrix')

# Display Loss vs. Accuracy Plot
if os.path.exists('loss_vs_accuracy.png'):
    st.image(Image.open('loss_vs_accuracy.png'), caption='Loss vs Predicted Values')
