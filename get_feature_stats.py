import pickle
import pandas as pd
from pathlib import Path

BASE_DIR = Path("D:/Downloads/credit_risk")
MODELS_DIR = BASE_DIR / "models"

with open(MODELS_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

features_to_check = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", 
    "CREDIT_INCOME_RATIO", "CREDIT_TERM", "DAYS_EMPLOYED", 
    "AMT_CREDIT", "AMT_ANNUITY", "LATE_PAYMENT_RATIO"
]

print("Feature stats:")
for f in features_to_check:
    if f in X_test.columns:
        col = X_test[f]
        print(f"{f}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}, median={col.median():.4f}")
    else:
        print(f"{f} not found in X_test columns!")
