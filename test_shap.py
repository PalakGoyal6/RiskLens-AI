import pickle
import joblib
import pandas as pd
import numpy as np
import shap
import json
from pathlib import Path

BASE_DIR = Path("D:/Downloads/credit_risk")
MODELS_DIR = BASE_DIR / "models"

# Load model
model_path = MODELS_DIR / "home_credit_catboost.pkl"
model = joblib.load(model_path)
print("Model loaded successfully:", type(model))

# Load X_test
with open(MODELS_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
print("X_test loaded successfully:", X_test.shape)

# Create explainer
explainer = shap.TreeExplainer(model)
print("Explainer created successfully:", type(explainer))

# Try explaining a single instance
sample_instance = X_test.iloc[0:1]
print("Explaining single instance...")
shap_values = explainer(sample_instance)
print("SHAP values shape:", shap_values.shape)
print("Base value:", shap_values.base_values)
print("Values:", shap_values.values)
print("Data:", shap_values.data)
