import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("wine_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set the page title
st.title("🍷 Wine Quality Prediction App")

st.markdown("""
This app predicts the **quality of wine** based on physicochemical features.
""")

# Define input fields
def user_input_features():
    type_input = st.selectbox("Wine Type", ["Red", "White"])
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.slider("Residual Sugar", 0.0, 15.0, 1.9)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.045)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 70, 15)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 300, 46)
    density = st.slider("Density", 0.9900, 1.0050, 0.9978)
    pH = st.slider("pH", 2.8, 4.0, 3.0)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

    data = {
        "type": 0 if type_input == "Red" else 1,
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Scale the input
scaled_input = scaler.transform(input_df)

# Predict button
if st.button("Predict Quality"):
    prediction = model.predict(scaled_input)
    st.success(f"🍇 Predicted Wine Quality: **{prediction[0]:.2f}** (out of 10)")
