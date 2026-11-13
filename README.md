# Diabetes Risk Prediction (Machine Learning Project)

Predict whether a patient is at risk of diabetes from medical measurements.
**Dataset:** Pima Indians Diabetes (768 patients, 8 features) - same data as the Kaggle
"Pima Indians Diabetes Database". Target `Outcome`: 1 = diabetic, 0 = not diabetic.

## How to run
```
pip install -r requirements.txt
python run_all.py          # runs every step in order
```
Or run one file at a time in the order below.

## Files (one job per file)
| File | What it does | Syllabus unit |
|---|---|---|
| `config.py` | Paths, column names, random seed | - |
| `01_data_cleaning.py` | Load CSV, treat impossible 0s as missing, fill with median, drop duplicates | I |
| `02_eda.py` | Class balance, histograms, correlation heatmap, boxplots, IQR outliers | II |
| `03_feature_engineering.py` | Cap outliers, create AgeGroup / BMICategory / Glucose_BMI, stratified train-test split | I, II |
| `04_train_models.py` | Train Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes with scaling pipeline + 5-fold cross-validation | IV, VI |
| `05_test_evaluate.py` | Test on unseen data: accuracy, precision, recall, F1, ROC-AUC, confusion matrix | IV |
| `06_tuning_explainability.py` | GridSearchCV on Random Forest, feature importance, save final model | VI |
| `07_clustering_pca.py` | K-Means + elbow + silhouette + PCA to group patients | V |
| `predict.py` | Enter a patient's values, get risk probability | VI |
| `run_all.py` | Runs everything | - |

Folders: `data/` (raw + cleaned + train/test CSVs), `models/` (saved .pkl), `outputs/` (graphs + results).

## Key results (test set)
- Best models reach ROC-AUC of about 0.82-0.83, recall for diabetic patients about 0.76.
- Most important factors: Glucose, BMI (and Glucose x BMI), Age.
- Clustering finds 3 patient groups; one (younger, lower glucose) has only ~13% diabetes rate versus ~52% in the others.

## Simple explanations (for viva)
- **Why replace 0 with median?** A glucose or BMI of 0 is impossible, so it is missing data. Median is not affected by outliers.
- **Why stratified split?** Keeps the same diabetic percentage in train and test.
- **Why scaling inside a Pipeline?** The scaler learns only from training data, so no information leaks from the test set.
- **Why class_weight="balanced"?** Diabetics are only ~35% of the data; this stops the model ignoring them.
- **Why recall matters?** Missing a sick patient (false negative) is worse than a false alarm.
- **Why cross-validation?** One split can be lucky; 5 folds give a more reliable score.
- **Why GridSearchCV?** Tries many hyperparameter combinations automatically and picks the best one.

## Limitations and ethics
- Small dataset (768 rows), all female patients of Pima heritage aged 21+, so it may not generalise to other groups.
- It is a learning project, not a medical tool. Predictions must never replace a doctor.
- Patient health data is sensitive: privacy and consent matter in real systems.
