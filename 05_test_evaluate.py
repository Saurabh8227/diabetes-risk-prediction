"""STEP 5 - TESTING / EVALUATION  (Unit IV)
Load saved models and check them on test.csv (data never seen in training).
Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
"""
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report)
from config import *

test = pd.read_csv(TEST_FILE)
X_test, y_test = test.drop(columns=TARGET), test[TARGET]

names = ["Logistic Regression", "KNN", "Decision Tree",
         "Random Forest", "SVM", "Naive Bayes"]
rows = []
fig, ax = plt.subplots(figsize=(7, 6))
for name in names:
    model = joblib.load(f"{MODEL_DIR}/{name.replace(' ', '_')}.pkl")
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC-AUC": roc_auc_score(y_test, proba)})
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)

ax.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves (test data)")
plt.savefig(f"{OUT_DIR}/roc_curves.png"); plt.close()

results = pd.DataFrame(rows).set_index("Model").round(3)
print(results.sort_values("ROC-AUC", ascending=False))
results.to_csv(f"{OUT_DIR}/model_comparison.csv")

# Confusion matrix + report for the best model
best = open(f"{MODEL_DIR}/best_model_name.txt").read()
model = joblib.load(f"{MODEL_DIR}/{best.replace(' ', '_')}.pkl")
pred = model.predict(X_test)
print(f"\n=== Detailed report: {best} ===")
print(classification_report(y_test, pred, target_names=["No Diabetes", "Diabetes"]))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                       display_labels=["No Diabetes", "Diabetes"]).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best}")
plt.savefig(f"{OUT_DIR}/confusion_matrix.png"); plt.close()
print("Saved plots + model_comparison.csv in outputs/")
