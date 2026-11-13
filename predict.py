"""STEP 8 - USE THE MODEL: enter patient values, get a risk prediction.
Run: python predict.py
"""
import joblib
import pandas as pd
from config import *

saved = joblib.load(f"{MODEL_DIR}/final_model.pkl")
model, features = saved["model"], saved["features"]

# Raw values for one patient
p = {"Pregnancies": 2, "Glucose": 150, "BloodPressure": 80, "SkinThickness": 30,
     "Insulin": 120, "BMI": 33.0, "DiabetesPedigree": 0.6, "Age": 45}

# Same engineered features as in step 3
p["AgeGroup"] = 0 if p["Age"] <= 30 else 1 if p["Age"] <= 45 else 2 if p["Age"] <= 60 else 3
b = p["BMI"]
p["BMICategory"] = 0 if b <= 18.5 else 1 if b <= 25 else 2 if b <= 30 else 3
p["Glucose_BMI"] = p["Glucose"] * p["BMI"] / 100

X = pd.DataFrame([p])[features]
prob = model.predict_proba(X)[0, 1]
print(f"Diabetes risk probability: {prob:.1%}")
print("Result:", "HIGH RISK - consult a doctor" if prob >= 0.5 else "Low risk")
