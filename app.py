# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:01:10 2025

@author: donad
"""

# app.py – Your Streamlit Web App

import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🫀", layout="centered")

# Title
st.title("🫀 Stroke Risk Prediction App")
st.markdown("""
This app predicts the likelihood of a stroke based on health and lifestyle factors.
Built using **Random Forest** and real health data.
""")

# Load model and columns
@st.cache_resource
def load_model():
    model = joblib.load('stroke_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_model()
    st.sidebar.success("✅ Model loaded!")
except Exception as e:
    st.error("❌ Could not load model. Make sure 'stroke_model.pkl' and 'model_columns.pkl' exist.")
    st.stop()

# User input form
st.subheader("📝 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 0, 100, 50)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    avg_glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 80.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with col2:
    gender = st.selectbox("Gender", ["Female", "Male"])
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Prepare input
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# One-hot encode
input_encoded = pd.get_dummies(input_data, columns=[
    'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
])

# Align with training columns
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# Predict
prediction = model.predict(input_encoded)[0]
probability = model.predict_proba(input_encoded)[0][1]

# Display result
st.write("---")
st.subheader("🎯 Prediction Result")

if prediction == 1:
    st.error(f"🚨 High Risk of Stroke")
    st.metric("Stroke Probability", f"{probability * 100:.2f}%")
else:
    st.success(f"✅ Low Risk of Stroke")
    st.metric("Stroke Probability", f"{probability * 100:.2f}%")

# Show feature importance
st.write("---")
if st.checkbox("📊 Show Feature Importance"):
    st.markdown("### Top Features Influencing Prediction")
    importances = model.feature_importances_
    feat_df = pd.Series(importances, index=model_columns).sort_values(ascending=False).head(8)
    st.bar_chart(feat_df)

# Footer
st.write("---")
st.markdown("💡 **Note**: This is a demo for educational purposes — not a substitute for medical advice.")