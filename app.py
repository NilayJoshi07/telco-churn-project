import streamlit as st
import pandas as pd
import joblib

# 1. Load the Model and Features
model = joblib.load('churn_model.pkl')
features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📞")

st.title("📞 Customer Retention Dashboard")
st.markdown("Use the sidebar to adjust customer details and predict churn probability.")

# 2. Sidebar for inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
fiber = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# 3. Predict Logic
input_data = pd.DataFrame(0, index=[0], columns=features)
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly
input_data['TotalCharges'] = tenure * monthly

if contract == "One year":
    input_data['Contract_One year'] = 1
elif contract == "Two year":
    input_data['Contract_Two year'] = 1

if fiber == "Fiber optic":
    for col in features:
        if "Fiber optic" in col:
            input_data[col] = 1

# 4. Show Result
prob = model.predict_proba(input_data)[0][1]

st.metric(label="Churn Probability", value=f"{prob:.1%}")

if prob > 0.5:
    st.error("🚨 **High Risk:** This customer is likely to churn.")
else:
    st.success("✅ **Low Risk:** This customer is likely to stay.")
