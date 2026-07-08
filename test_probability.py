import pickle
import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path

BASE_DIR = Path("D:/Downloads/credit_risk")
MODELS_DIR = BASE_DIR / "models"

# Load model and data
model = joblib.load(MODELS_DIR / "home_credit_catboost.pkl")
with open(MODELS_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

explainer = shap.TreeExplainer(model)
sample = X_test.iloc[0:5]

# SHAP values
shap_res = explainer(sample)
for i in range(5):
    logit_sum = shap_res.base_values[i] + shap_res.values[i].sum()
    prob_from_logit = 1 / (1 + np.exp(-logit_sum))
    model_prob = model.predict_proba(sample.iloc[i:i+1])[0, 1]
    print(f"Sample {i}: Logit Sum = {logit_sum:.4f}, Prob from Logit = {prob_from_logit:.4f}, Model Prob = {model_prob:.4f}")
