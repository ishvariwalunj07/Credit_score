import streamlit as st # type: ignore
import numpy as np
import joblib # type: ignore

# Load your trained model
model = joblib.load('credit_model.pkl')

st.title("💳 Credit Scoring Predictor")
st.markdown("Enter applicant details to predict credit risk.")

# Input form (Only 3 features used in training)
income = st.number_input("Monthly Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History (0 = Bad, 1 = Good)", [0, 1])

# Predict on button click
if st.button("Predict Credit Risk"):
    input_data = np.array([[income, loan_amount, credit_history]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"❌ High Risk: {probability*100:.2f}% chance of default.")
    else:
        st.success(f"✅ Low Risk: {probability*100:.2f}% chance of default.")
