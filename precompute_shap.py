import pickle
import joblib
import pandas as pd
import numpy as np
import shap
import json
from pathlib import Path

BASE_DIR = Path("D:/Downloads/credit_risk")
MODELS_DIR = BASE_DIR / "models"

# Load model and data
print("Loading model and data...")
model = joblib.load(MODELS_DIR / "home_credit_catboost.pkl")
with open(MODELS_DIR / "X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open(MODELS_DIR / "y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

print("X_test shape:", X_test.shape)

# Create explainer
print("Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# We will sample 1000 instances for global SHAP calculation
print("Sampling 1000 instances...")
sample_df = X_test.sample(n=1000, random_state=42)
sample_y = y_test.loc[sample_df.index]

# Compute SHAP values
print("Computing SHAP values for the sample...")
shap_values = explainer(sample_df)

# Prepare cache data
# We need:
# - Feature names
# - SHAP values array
# - Base value (expected value)
# - Feature values data
# - Index (applicant IDs)
# - Actual targets

# Calculate global importance
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

top_features = importance_df['feature'].tolist()
print("Top 15 features:")
for i, row in importance_df.head(15).iterrows():
    print(f"- {row['feature']}: {row['importance']:.4f}")

# Save SHAP cache as a pickle file (since numpy arrays are not directly JSON-serializable without casting)
cache = {
    'base_value': float(shap_values.base_values[0]),
    'feature_names': list(X_test.columns),
    'shap_values': shap_values.values,
    'data': sample_df.values,
    'ids': list(sample_df.index),
    'targets': list(sample_y.values),
    'top_features': top_features
}

cache_path = MODELS_DIR / "shap_cache.pkl"
with open(cache_path, "wb") as f:
    pickle.dump(cache, f)

print(f"Precomputed SHAP cache saved to {cache_path}")
