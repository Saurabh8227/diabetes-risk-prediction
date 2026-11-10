"""STEP 3 - FEATURE ENGINEERING + TRAIN/TEST SPLIT  (Unit I & II)
- cap outliers, create new features, split data (stratified)
- scaling is done later inside the model pipeline (avoids data leakage)
Output: data/train.csv, data/test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *

df = pd.read_csv(CLEAN_FILE)

# 1. Cap outliers at IQR limits (winsorizing) instead of deleting rows
for col in df.columns.drop(TARGET):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# 2. New features
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 30, 45, 60, 120],
                        labels=[0, 1, 2, 3]).astype(int)       # encoded as numbers
df["BMICategory"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                           labels=[0, 1, 2, 3]).astype(int)     # under/normal/over/obese
df["Glucose_BMI"] = df["Glucose"] * df["BMI"] / 100

df.to_csv(FEATURES_FILE, index=False)

# 3. Stratified split keeps the same diabetic % in train and test
train, test = train_test_split(df, test_size=TEST_SIZE,
                               stratify=df[TARGET], random_state=RANDOM_STATE)
train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)

print("Features:", list(df.columns.drop(TARGET)))
print("Train shape:", train.shape, "| Test shape:", test.shape)
print("Diabetic % train:", round(train[TARGET].mean() * 100, 1),
      "| test:", round(test[TARGET].mean() * 100, 1))
