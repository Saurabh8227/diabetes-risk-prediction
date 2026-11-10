"""STEP 2 - EXPLORATORY DATA ANALYSIS  (Unit II)
Plots to understand the data. All images saved in outputs/.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

df = pd.read_csv(CLEAN_FILE)
print(df.describe().round(2))

# 1. Class balance (how many diabetic vs non-diabetic)
print("\nClass counts:\n", df[TARGET].value_counts())
sns.countplot(x=TARGET, data=df)
plt.title("Diabetes (1) vs No Diabetes (0)")
plt.savefig(f"{OUT_DIR}/eda_class_balance.png"); plt.close()

# 2. Histograms of all features
df.drop(columns=TARGET).hist(figsize=(12, 8), bins=20)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/eda_histograms.png"); plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(f"{OUT_DIR}/eda_correlation.png"); plt.close()
print("\nCorrelation with Outcome:\n",
      df.corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

# 4. Boxplots (outlier detection visually)
df.drop(columns=TARGET).plot(kind="box", subplots=True, layout=(2, 4),
                             figsize=(14, 7), sharex=False, sharey=False)
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/eda_boxplots.png"); plt.close()

# 5. Outlier count using the IQR rule
print("\nOutliers per column (IQR rule):")
for col in df.columns.drop(TARGET):
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col:18s}: {n}")

# 6. Glucose vs Outcome
sns.boxplot(x=TARGET, y="Glucose", data=df)
plt.title("Glucose by Outcome")
plt.savefig(f"{OUT_DIR}/eda_glucose_vs_outcome.png"); plt.close()
print("\nEDA plots saved in outputs/")
