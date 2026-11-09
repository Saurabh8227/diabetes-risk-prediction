"""Shared settings used by every script (change paths/seed here only)."""
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_FILE = os.path.join(DATA_DIR, "raw.csv")
CLEAN_FILE = os.path.join(DATA_DIR, "clean.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"]
# In these columns a value of 0 is impossible, so 0 really means "missing"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

TARGET = "Outcome"
RANDOM_STATE = 42
TEST_SIZE = 0.2

for folder in (DATA_DIR, OUT_DIR, MODEL_DIR):
    os.makedirs(folder, exist_ok=True)
