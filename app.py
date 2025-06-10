
import streamlit as st
import pandas as pd
import joblib

# Load trained model and test data
model = joblib.load("xgb_model.joblib")
X_test = pd.read_csv("X_test_sample.csv")

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Real-Time Fraud Detection")

# Slider to select transaction
index = st.slider("Pick a transaction to evaluate", 0, len(X_test) - 1, 0)
transaction = X_test.iloc[[index]]

# Show transaction details
st.subheader("Transaction Features")
st.dataframe(transaction.T)

# Predict
prediction = model.predict(transaction)[0]
proba = model.predict_proba(transaction)[0][1]

# Display result
st.markdown("### 🔍 Prediction Result:")
st.success("✅ Legitimate Transaction" if prediction == 0 else "⚠️ Fraudulent Transaction")
st.metric("Fraud Probability", f"{proba:.4f}")
