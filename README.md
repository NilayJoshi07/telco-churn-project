# telco-churn-project
An end-to-end Machine Learning pipeline and interactive dashboard to predict telecommunications churn. Built with Python and Random Forest; deployed via Streamlit.

Project Overview:
This project addresses the "Churn" problem in the telecommunications industry—where customers switch to competitors. By analyzing historical data, I built a predictive engine that allows businesses to:
Identify high-risk customers based on behavior.
Quantify the probability of a customer leaving.
Simulate how changing contract types or monthly charges affects retention.

Interactive Features:
The deployed Streamlit app features a dynamic sidebar where you can adjust.
Tenure: How many months the customer has been with the company.
Contract Type: Month-to-month, One year, or Two year.
Internet Service: Fiber optic, DSL, or None.
Monthly Charges: Financial impact on churn risk.

Model Performance & Results:
The core of this project is a Random Forest Classifier. During training, the model achieved the following:
Metric Score Accuracy: 80%-Key Indicators Tenure and Contract Type were the strongest predictors of loyalty.
Insight: Customers on "Month-to-month" contracts with "Fiber Optic" service showed the highest probability of churn, suggesting a need for better long-term incentives in those segments.

Features
What-If Analysis: Live simulation of customer behavior changes.
Feature Sensitivity: Visual breakdown of the top 3 factors driving individual churn risk.
Automated Retention Strategy: Calculates the exact price discount needed to retain high-risk customers.
Market Benchmarking: Compares individual profiles against global loyal customer averages.

Tech Stack:
Python 3.14
Machine Learning: Scikit-Learn (Random Forest)
Web Framework: Streamlit
Data Processing: Pandas & NumPy
Model Export: Joblib
Visualization: seaborn & matplotlib

Repository Structure
app.py: The main script for the web application.
churn_model.pkl: The trained Random Forest model.
model_features.pkl: Saved feature mapping for consistent predictions.

requirements.txt: Configuration for Streamlit Cloud.
