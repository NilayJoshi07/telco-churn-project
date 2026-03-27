import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Model, Features, and the new Insights
model = joblib.load('churn_model.pkl')
features = joblib.load('model_features.pkl')
stats = joblib.load('statistical_insights.pkl')

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📞", layout="wide")

# --- Helper Functions (The "Brain" of the Deep Dive) ---
def get_feature_sensitivity(model, input_df):
    base_prob = model.predict_proba(input_df)[0][1]
    sensitivity = []
    for col in input_df.columns:
        temp_df = input_df.copy()
        temp_df[col] = 0 
        new_prob = model.predict_proba(temp_df)[0][1]
        sensitivity.append({'Feature': col, 'Impact': base_prob - new_prob})
    return pd.DataFrame(sensitivity).sort_values(by='Impact', ascending=False).head(3)

def find_churn_threshold(model, input_df):
    temp_df = input_df.copy()
    original_charge = temp_df['MonthlyCharges'].values[0]
    current_charge = original_charge
    while current_charge > 10:
        if model.predict_proba(temp_df)[0][1] < 0.5:
            break
        current_charge -= 1.0
        temp_df['MonthlyCharges'] = current_charge
    return current_charge, original_charge - current_charge

# --- UI Layout ---
st.title("📞 Customer Retention Dashboard")
st.markdown("Adjust customer details in the sidebar to simulate **What-If** scenarios.")

# 2. Sidebar for inputs (The What-If Scenario)
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly = st.sidebar.slider("Monthly Charges ($)", 18, 120, 70)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
fiber = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

# 3. Prepare Input Data
input_data = pd.DataFrame(0, index=[0], columns=features)
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly
input_data['TotalCharges'] = tenure * monthly

if contract == "One year": input_data['Contract_One year'] = 1
elif contract == "Two year": input_data['Contract_Two year'] = 1

if fiber == "Fiber optic":
    for col in features:
        if "Fiber optic" in col: input_data[col] = 1

# 4. Predict Result
prob = model.predict_proba(input_data)[0][1]

# --- Display Results in Tabs ---
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🔍 Risk Analysis", "🛠️ Retention Strategy"])

with tab1:
    st.metric(label="Churn Probability", value=f"{prob:.1%}")
    if prob > 0.5:
        st.error("🚨 **High Risk:** This customer is likely to churn.")
    else:
        st.success("✅ **Low Risk:** This customer is likely to stay.")
    
    st.divider()
    st.subheader("📊 Market Benchmarks")
    st.write(f"Average Monthly Charge for Loyal Customers: **${stats['avg_monthly_charges']:.2f}**")

with tab2:
    st.subheader("Why is this customer at risk?")
    sensitivity_df = get_feature_sensitivity(model, input_data)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Impact', y='Feature', data=sensitivity_df, palette='Reds_r')
    plt.title("Top Factors Increasing Churn Risk")
    st.pyplot(fig)
    st.info("The factors above show which customer traits are pushing the probability higher.")

with tab3:
    st.subheader("Personalized Retention Plan")
    if prob > 0.5:
        safe_price, discount = find_churn_threshold(model, input_data)
        st.warning(f"**Financial Intervention Recommended:**")
        st.write(f"- Offer a monthly discount of: **${discount:.2f}**")
        st.write(f"- New target monthly charge: **${safe_price:.2f}**")
    else:
        st.success("Customer is currently stable. No defensive discount needed.")