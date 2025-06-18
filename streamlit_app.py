
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("xgb_best_model.pkl")

st.set_page_config(page_title="Credit Default Risk Prediction")
st.title("🏦 Credit Default Risk Predictor")

st.sidebar.header("Input Applicant Info")

income = st.sidebar.number_input("Annual Income", min_value=1000, value=120000)
credit = st.sidebar.number_input("Loan Amount", min_value=1000, value=250000)
annuity = st.sidebar.number_input("Annuity", min_value=100, value=25000)
goods_price = st.sidebar.number_input("Goods Price", min_value=1000, value=220000)
ext_source_2 = st.sidebar.slider("External Source 2", 0.0, 1.0, 0.6)
ext_source_3 = st.sidebar.slider("External Source 3", 0.0, 1.0, 0.5)

credit_income_ratio = credit / income if income else 0
annuity_income_ratio = annuity / income if income else 0

input_df = pd.DataFrame([{
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit,
    "AMT_ANNUITY": annuity,
    "AMT_GOODS_PRICE": goods_price,
    "EXT_SOURCE_2": ext_source_2,
    "EXT_SOURCE_3": ext_source_3,
    "CREDIT_INCOME_RATIO": credit_income_ratio,
    "ANNUITY_INCOME_RATIO": annuity_income_ratio
}])

st.subheader("📋 Input Preview")
st.dataframe(input_df)

# Predict
prob = model.predict_proba(input_df)[0][1]
st.subheader(f"📈 Default Probability: **{prob:.2%}**")

# SHAP Explanation
st.subheader("🔍 Feature Importance (SHAP)")
explainer = shap.Explainer(model)
shap_values = explainer(input_df)
fig = shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
