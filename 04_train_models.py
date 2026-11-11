"""STEP 4 - TRAINING  (Unit IV & VI)
Train several models using only train.csv. Compare using 5-fold cross-validation.
Scaling is inside a Pipeline so it is learned only from training folds.
Output: models/<name>.pkl and models/best_model_name.txt
"""
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from config import *

train = pd.read_csv(TRAIN_FILE)
X_train, y_train = train.drop(columns=TARGET), train[TARGET]

# class_weight="balanced" handles the imbalance (fewer diabetic patients)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                            random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
for name, model in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"{name:20s} CV ROC-AUC: {scores.mean():.3f} (+/- {scores.std():.3f})")
    pipe.fit(X_train, y_train)                       # final fit on all training data
    joblib.dump(pipe, f"{MODEL_DIR}/{name.replace(' ', '_')}.pkl")

best = max(results, key=results.get)
open(f"{MODEL_DIR}/best_model_name.txt", "w").write(best)
print("\nBest model by cross-validation:", best)
