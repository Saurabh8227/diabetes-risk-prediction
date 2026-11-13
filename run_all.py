"""Run the whole pipeline in order:  python run_all.py"""
import subprocess, sys
steps = ["01_data_cleaning.py", "02_eda.py", "03_feature_engineering.py",
         "04_train_models.py", "05_test_evaluate.py",
         "06_tuning_explainability.py", "07_clustering_pca.py", "predict.py"]
for s in steps:
    print(f"\n{'='*20} {s} {'='*20}")
    if subprocess.run([sys.executable, s]).returncode != 0:
        sys.exit(f"Stopped: {s} failed")
