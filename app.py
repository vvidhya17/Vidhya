import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="AI Health Pro", layout="wide")

st.title("AI Based Multi Disease Prediction and Recommendation System")

# -------------------------------
# FUNCTION
# -------------------------------
def convert(v):
    return 0 if v=="LOW" else 1 if v=="MEDIUM" else 2

# -------------------------------
# INPUT
# -------------------------------
st.markdown("### Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 0, 100, 25)
    gender = st.selectbox("Gender", ["Male","Female"])
    bmi = st.selectbox("BMI", ["LOW","MEDIUM","HIGH"])

with col2:
    bp = st.selectbox("Blood Pressure", ["LOW","MEDIUM","HIGH"])
    glucose = st.selectbox("Glucose", ["LOW","MEDIUM","HIGH"])
    chol = st.selectbox("Cholesterol", ["LOW","MEDIUM","HIGH"])

with col3:
    activity = st.selectbox("Activity", ["LOW","MEDIUM","HIGH"])
    smoking = st.selectbox("Smoking", ["No","Yes"])
    alcohol = st.selectbox("Alcohol", ["No","Yes"])

fever = st.selectbox("Fever", ["No","Yes"])
chest = st.selectbox("Chest Pain", ["No","Yes"])

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
    model = RandomForestClassifier()
    model.fit(X, data[d])
    models[d] = model

# -------------------------------
# BUTTON
# -------------------------------
if st.button("🔍 Predict Result"):

    # Encode
    age_val = 0 if age < 30 else 1 if age < 60 else 2
    gender_val = 0 if gender=="Male" else 1

    input_df = pd.DataFrame([{
        'Age':age_val,
        'Gender':gender_val,
        'BMI':convert(bmi),
        'BP':convert(bp),
        'Glucose':convert(glucose),
        'Cholesterol':convert(chol),
        'Activity':convert(activity),
        'Smoking':1 if smoking=="Yes" else 0,
        'Alcohol':1 if alcohol=="Yes" else 0,
        'Symptoms':(1 if fever=="Yes" else 0) + (1 if chest=="Yes" else 0)
    }])

    # -------------------------------
    # PREDICTION
    # -------------------------------
    st.markdown("## Prediction Results")

    results = {}
    for d, m in models.items():
        prob = m.predict_proba(input_df)[0][1]
        results[d] = prob
        st.write(f"{d} → {round(prob,2)}")

    st.markdown("---")

    # -------------------------------
    # DASHBOARD (3 CHARTS)
    # -------------------------------
    st.markdown("## Health Dashboard")

    c1, c2, c3 = st.columns(3)

    # RADAR
    with c1:
        labels = list(results.keys())
        values = list(results.values())
        values += values[:1]

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig1 = plt.figure(figsize=(2.5,2.5))
        ax1 = fig1.add_subplot(111, polar=True)

        ax1.plot(angles, values)
        ax1.fill(angles, values, alpha=0.1)

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(labels, fontsize=5)
        ax1.set_yticks([])

        st.pyplot(fig1)
        plt.close(fig1)

    # 3D BAR
    with c2:
        fig2 = plt.figure(figsize=(2.5,2.5))
        ax2 = fig2.add_subplot(111, projection='3d')

        xs = np.arange(len(results))
        ys = [v*0.5 for v in results.values()]

        ax2.bar3d(xs, np.zeros(len(xs)), np.zeros(len(xs)),
                  0.3, 0.3, ys)

        ax2.set_xticks(xs)
        ax2.set_xticklabels(list(results.keys()), fontsize=5, rotation=25)
        ax2.set_zticks([])

        st.pyplot(fig2)
        plt.close(fig2)

    # LINE CHART
    with c3:
        labels = list(results.keys())
        values = list(results.values())

        fig3, ax3 = plt.subplots(figsize=(2.5,2.5))

        ax3.plot(labels, values, marker='o', color='black')

        for i, v in enumerate(values):
            ax3.text(i, v+0.03, f"{round(v,2)}", fontsize=6, ha='center')

        ax3.set_ylim(0,1.1)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, fontsize=5, rotation=25)
        ax3.set_yticks([])

        st.pyplot(fig3)
        plt.close(fig3)

    # -------------------------------
    # RECOMMENDATIONS
    # -------------------------------
    st.markdown("---")
    st.subheader("Recommendations")

    for disease, prob in results.items():
        if prob > 0.6:
            st.warning(f"High risk of {disease}")
        elif prob > 0.3:
            st.info(f"Moderate risk of {disease}")
        else:
            st.success(f"Low risk of {disease}")

    # Overall Score
    score = sum(results.values()) / len(results)
    st.metric("Overall Risk Score", round(score,2))