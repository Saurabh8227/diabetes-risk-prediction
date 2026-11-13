"""DASHBOARD - shows data, graphs, model accuracy and live prediction.
Run:  python -m streamlit run app.py
(first run python run_all.py once, so results and models exist)
"""
import os
import pandas as pd
import joblib
import streamlit as st
from config import *

st.set_page_config(page_title="Diabetes Risk Dashboard", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Risk Prediction Dashboard")
st.caption("Pima Indians Diabetes dataset | Machine Learning project")

if not os.path.exists(f"{OUT_DIR}/model_comparison.csv"):
    st.error("Results not found. Run `python run_all.py` first, then reload this page.")
    st.stop()

def show(img, caption=None):
    path = f"{OUT_DIR}/{img}"
    if os.path.exists(path):
        st.image(path, caption=caption, width="stretch")

raw = pd.read_csv(CLEAN_FILE)
results = pd.read_csv(f"{OUT_DIR}/model_comparison.csv").set_index("Model")
best_name = results["ROC-AUC"].idxmax()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔍 EDA", "🤖 Model Results", "👥 Patient Groups", "🧪 Predict"])

# ---------- Overview ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(raw))
    c2.metric("Features", raw.shape[1] - 1)
    c3.metric("Diabetic %", f"{raw[TARGET].mean()*100:.1f}%")
    c4.metric("Best model (ROC-AUC)", best_name, f"{results.loc[best_name,'ROC-AUC']:.3f}")
    st.subheader("Cleaned data (first 10 rows)")
    st.dataframe(raw.head(10), width="stretch")
    st.subheader("Summary statistics")
    st.dataframe(raw.describe().round(2), width="stretch")

# ---------- EDA ----------
with tab2:
    a, b = st.columns(2)
    with a:
        show("eda_class_balance.png", "Class balance")
        show("eda_correlation.png", "Correlation heatmap")
    with b:
        show("eda_glucose_vs_outcome.png", "Glucose vs Outcome")
        show("eda_boxplots.png", "Boxplots (outliers)")
    show("eda_histograms.png", "Feature distributions")

# ---------- Model results ----------
with tab3:
    st.subheader("Accuracy and metrics on unseen test data")
    st.dataframe(results.style.highlight_max(axis=0, color="#1f6f43")
                 .format("{:.3f}"), width="stretch")
    st.bar_chart(results[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]])
    a, b = st.columns(2)
    with a:
        show("roc_curves.png", "ROC curves")
    with b:
        show("confusion_matrix.png", "Confusion matrix (best CV model)")
    st.subheader("Which features matter most?")
    show("feature_importance.png")
    st.info("Recall = how many diabetic patients the model catches. "
            "In healthcare, missing a sick patient is worse than a false alarm.")

# ---------- Clusters ----------
with tab4:
    a, b = st.columns(2)
    with a:
        show("clusters_pca.png", "Patient clusters (PCA view)")
    with b:
        show("cluster_k_selection.png", "Choosing k (elbow + silhouette)")

# ---------- Predict ----------
with tab5:
    st.subheader("Enter patient details")
    model_path = f"{MODEL_DIR}/final_model.pkl"
    if not os.path.exists(model_path):
        st.warning("final_model.pkl not found. Run python run_all.py first.")
    else:
        saved = joblib.load(model_path)
        model, features = saved["model"], saved["features"]
        c1, c2, c3, c4 = st.columns(4)
        preg = c1.number_input("Pregnancies", 0, 20, 2)
        glu = c2.number_input("Glucose", 40, 250, 150)
        bp = c3.number_input("Blood Pressure", 20, 130, 80)
        skin = c4.number_input("Skin Thickness", 5, 100, 30)
        ins = c1.number_input("Insulin", 10, 900, 120)
        bmi = c2.number_input("BMI", 10.0, 70.0, 33.0)
        dpf = c3.number_input("Diabetes Pedigree", 0.05, 2.5, 0.6)
        age = c4.number_input("Age", 18, 100, 45)

        if st.button("Predict risk", type="primary"):
            p = {"Pregnancies": preg, "Glucose": glu, "BloodPressure": bp,
                 "SkinThickness": skin, "Insulin": ins, "BMI": bmi,
                 "DiabetesPedigree": dpf, "Age": age}
            p["AgeGroup"] = 0 if age <= 30 else 1 if age <= 45 else 2 if age <= 60 else 3
            p["BMICategory"] = 0 if bmi <= 18.5 else 1 if bmi <= 25 else 2 if bmi <= 30 else 3
            p["Glucose_BMI"] = glu * bmi / 100
            prob = model.predict_proba(pd.DataFrame([p])[features])[0, 1]
            st.progress(float(prob))
            st.metric("Diabetes risk probability", f"{prob:.1%}")
            if prob >= 0.5:
                st.error("HIGH RISK - please consult a doctor.")
            else:
                st.success("Low risk.")
            st.caption("Learning project only - not a medical diagnosis.")
