import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# Correct model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

# Load model safely
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load scaler safely
with open(SCALER_PATH, "rb") as s:
    scaler = pickle.load(s)


# Page settings
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="centered",
)

# THEME-AWARE CSS (auto adapts to dark/light)
st.markdown("""
<style>

:root {
    --bg: transparent;
    --card-bg: rgba(255,255,255,0.1); 
    --text-color: inherit;
    --shadow: none;
}

/* Light mode */
[data-theme="light"] {
    --bg: #f6f9fc;
    --card-bg: #ffffff;
    --text-color: #1e1e1e;
    --shadow: 0px 4px 12px rgba(0,0,0,0.10);
}

/* Dark mode */
[data-theme="dark"] {
    --bg: #0e1117;
    --card-bg: #161b22;
    --text-color: #f5f5f5;
    --shadow: 0px 4px 12px rgba(0,0,0,0.50);
}

.main {
    background-color: var(--bg) !important;
}

/* Remove the unwanted extra spacing below subtitle */
.subtitle-text {
    margin-bottom: 5px !important;
    padding-bottom: 0px !important;
}

.block-container {
    padding-top: 4rem !important;
}

.title-text {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: -5px;
    color: var(--text-color);
}

.subtitle-text {
    text-align: center;
    font-size: 16px;
    color: var(--text-color);
    opacity: 0.85;
}

/* Prediction result card */
.stCard {
    background: var(--card-bg);
    padding: 25px;
    border-radius: 18px;
    box-shadow: var(--shadow);
    color: var(--text-color);
}

/* Button styling */
.predict-btn button {
    border-radius: 10px !important;
    font-size: 18px !important;
    padding: 10px 20px !important;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h2 class='title-text'>🏥 Medical Insurance Cost Predictor</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Predict your yearly medical insurance cost instantly.</p>", unsafe_allow_html=True)

# 🔹 INPUT FIELDS (no wrapper card — no empty box)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    children = st.number_input("Children", min_value=0, max_value=10, step=1)

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Predict button
st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
predict = st.button("🔍 Predict Insurance Cost")
st.markdown("</div>", unsafe_allow_html=True)

# 🔹 Prediction Logic
if predict:
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    input_df['sex'] = input_df['sex'].map({'male': 1, 'female': 0})
    input_df['smoker'] = input_df['smoker'].map({"yes": 1, 'no': 0})
    input_df['region'] = input_df['region'].map({
        'southeast': 0, "southwest": 1, 'northwest': 2, "northeast": 3
    })

    num_cols = ['age', 'bmi', 'children']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]

    # Result card — only appears on prediction
    st.markdown(f"""
    <div class='stCard' style='text-align:center; margin-top:20px;'>
        <h3 style='color: var(--text-color);'>💰 Estimated Annual Charge</h3>
        <p style='font-size:30px; font-weight:700; color: var(--text-color);'>
            ${prediction:,.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)


