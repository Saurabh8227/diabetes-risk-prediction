"""STEP 6 - TUNING + EXPLAINABILITY  (Unit VI)
- GridSearchCV on Random Forest (recall-focused, because missing a sick
  patient is worse than a false alarm)
- Feature importance to explain which factors drive the prediction
Output: models/final_model.pkl
"""
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance
from config import *

train, test = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
X_train, y_train = train.drop(columns=TARGET), train[TARGET]
X_test, y_test = test.drop(columns=TARGET), test[TARGET]

# 1. Hyperparameter tuning
grid = {"n_estimators": [100, 200],
        "max_depth": [3, 5, 8],
        "min_samples_leaf": [1, 5, 10]}
search = GridSearchCV(RandomForestClassifier(class_weight="balanced",
                                             random_state=RANDOM_STATE),
                      grid, cv=5, scoring="roc_auc", n_jobs=-1)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best CV ROC-AUC:", round(search.best_score_, 3))

# 2. Evaluate tuned model on test data
final = search.best_estimator_
pred = final.predict(X_test)
print("\nTest ROC-AUC:", round(roc_auc_score(y_test, final.predict_proba(X_test)[:, 1]), 3))
print(classification_report(y_test, pred, target_names=["No Diabetes", "Diabetes"]))

# 3. Save final model
joblib.dump({"model": final, "features": list(X_train.columns)},
            f"{MODEL_DIR}/final_model.pkl")

# 4. Explainability: which features matter most?
imp = pd.Series(final.feature_importances_, index=X_train.columns).sort_values()
imp.plot(kind="barh", figsize=(8, 5), title="Feature Importance (Random Forest)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/feature_importance.png"); plt.close()
print("\nTop 5 features:\n", imp.sort_values(ascending=False).head(5).round(3))

perm = permutation_importance(final, X_test, y_test, n_repeats=10,
                              random_state=RANDOM_STATE, scoring="roc_auc")
print("\nPermutation importance (test):\n",
      pd.Series(perm.importances_mean, index=X_test.columns)
      .sort_values(ascending=False).head(5).round(3))
