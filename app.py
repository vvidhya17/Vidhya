import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Health Pro", layout="wide")

st.title("AI Based Multi Disease Prediction and Recommendation System")

# -------------------------------
# SIDEBAR FILTERS (Power BI style)
# -------------------------------
st.sidebar.header("Patient Filters")

age = st.sidebar.slider("Age", 0, 100, 25)
gender = st.sidebar.selectbox("Gender", ["Male","Female"])
bmi = st.sidebar.selectbox("BMI", ["LOW","MEDIUM","HIGH"])
bp = st.sidebar.selectbox("Blood Pressure", ["LOW","MEDIUM","HIGH"])

glucose = st.sidebar.selectbox("Glucose", ["LOW","MEDIUM","HIGH"])
chol = st.sidebar.selectbox("Cholesterol", ["LOW","MEDIUM","HIGH"])
activity = st.sidebar.selectbox("Activity", ["LOW","MEDIUM","HIGH"])

smoking = st.sidebar.selectbox("Smoking", ["No","Yes"])
alcohol = st.sidebar.selectbox("Alcohol", ["No","Yes"])

fever = st.sidebar.selectbox("Fever", ["No","Yes"])
chest = st.sidebar.selectbox("Chest Pain", ["No","Yes"])

predict_btn = st.sidebar.button("🔍 Predict Result")

# -------------------------------
# CONVERT FUNCTION
# -------------------------------
def convert(v):
    return 0 if v=="LOW" else 1 if v=="MEDIUM" else 2

# -------------------------------
# DATA
# -------------------------------
np.random.seed(1)

data = pd.DataFrame({
    'Age': np.random.randint(0,3,800),
    'Gender': np.random.randint(0,2,800),
    'BMI': np.random.randint(0,3,800),
    'BP': np.random.randint(0,3,800),
    'Glucose': np.random.randint(0,3,800),
    'Cholesterol': np.random.randint(0,3,800),
    'Activity': np.random.randint(0,3,800),
    'Smoking': np.random.randint(0,2,800),
    'Alcohol': np.random.randint(0,2,800),
    'Symptoms': np.random.randint(0,3,800)
})

data['Diabetes'] = (data['Glucose']==2).astype(int)
data['Heart'] = ((data['BP']==2)&(data['Cholesterol']==2)).astype(int)
data['Hypertension'] = (data['BP']==2).astype(int)
data['Obesity'] = (data['BMI']==2).astype(int)
data['Stroke'] = ((data['Smoking']==1)&(data['Heart']==1)).astype(int)
data['Liver'] = ((data['Alcohol']==1)&(data['Cholesterol']==2)).astype(int)

X = data.drop(columns=['Diabetes','Heart','Hypertension','Obesity','Stroke','Liver'])

models = {}
for d in ['Diabetes','Heart','Hypertension','Obesity','Stroke','Liver']:
    m = RandomForestClassifier()
    m.fit(X, data[d])
    models[d] = m

# -------------------------------
# PREDICTION
# -------------------------------
if predict_btn:

    # Age conversion
    if age < 30:
        age_val = 0
    elif age < 60:
        age_val = 1
    else:
        age_val = 2

    gender_val = 0 if gender=="Male" else 1
    bmi_val = convert(bmi)
    bp_val = convert(bp)
    glucose_val = convert(glucose)
    chol_val = convert(chol)
    activity_val = convert(activity)

    smoking_val = 1 if smoking=="Yes" else 0
    alcohol_val = 1 if alcohol=="Yes" else 0

    symptoms_val = (1 if fever=="Yes" else 0) + (1 if chest=="Yes" else 0)

    input_df = pd.DataFrame([{
        'Age':age_val,'Gender':gender_val,'BMI':bmi_val,'BP':bp_val,'Glucose':glucose_val,
        'Cholesterol':chol_val,'Activity':activity_val,'Smoking':smoking_val,
        'Alcohol':alcohol_val,'Symptoms':symptoms_val
    }])

    # -------------------------------
    # RESULTS
    # -------------------------------
    st.subheader("Prediction Results")

    results = {}
    for d,m in models.items():
        p = m.predict_proba(input_df)[0][1]
        results[d] = p
        st.write(f"{d} → {round(p,2)}")

    # -------------------------------
    # KPI CARDS
    # -------------------------------
    st.markdown("### Key Health Indicators")

    c1, c2, c3 = st.columns(3)
    c1.metric("Diabetes Risk", round(results['Diabetes'],2))
    c2.metric("Heart Risk", round(results['Heart'],2))
    c3.metric("Stroke Risk", round(results['Stroke'],2))

    st.markdown("---")

    # -------------------------------
    # DASHBOARD GRID
    # -------------------------------
    st.markdown("## Health Dashboard")

    # Row 1
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Risk Comparison")
        st.bar_chart(results)

    with colB:
        st.subheader("Risk Distribution")
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.pie(results.values(), labels=None, autopct='%1.1f%%')
        ax2.legend(results.keys(), loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
        st.pyplot(fig2)

    # Row 2
    colC, colD = st.columns(2)

    with colC:
        st.subheader("BMI & Glucose Distribution")
        fig5, ax5 = plt.subplots(figsize=(4,3))
        sns.violinplot(data=data[['BMI','Glucose']], ax=ax5)
        st.pyplot(fig5)

    with colD:
        st.subheader("Correlation Matrix")
        fig4, ax4 = plt.subplots(figsize=(4,3))
        sns.heatmap(data.corr(), cmap='coolwarm', annot=False, ax=ax4)
        st.pyplot(fig4)

    # -------------------------------
    # RECOMMENDATIONS
    # -------------------------------
    st.markdown("---")
    st.subheader("Recommendations")

    for disease, prob in results.items():
        if prob > 0.6:
            st.warning(f"High risk of {disease} - Consult doctor")
        elif prob > 0.3:
            st.info(f"Moderate risk of {disease}")
        else:
            st.success(f"Low risk of {disease}")

    score = sum(results.values())/len(results)
    st.metric("Overall Risk Score", round(score,2))