"""STEP 1 - DATA CLEANING  (Unit I)
Load the raw CSV, fix impossible zeros, fill missing values, remove duplicates.
Output: data/clean.csv
"""
import numpy as np
import pandas as pd
from config import *

# 1. Load data (raw file has no header row)
df = pd.read_csv(RAW_FILE, header=None, names=COLUMNS)
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.info())

# 2. Zeros in some columns are really missing values -> convert to NaN
df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
print("\nMissing values per column:\n", df.isnull().sum())

# 3. Fill missing values with the MEDIAN (robust to outliers)
for col in ZERO_AS_MISSING:
    df[col] = df[col].fillna(df[col].median())

# 4. Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed: {before - len(df)}")

# 5. Save
df.to_csv(CLEAN_FILE, index=False)
print("Clean data saved to", CLEAN_FILE)
