import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from nb_common import md, code, save, SETUP_SNIPPET

cells = []

cells.append(md('''# Model Training & Hyperparameter Tuning

This notebook loads the engineered dataset and feature list, trains several machine learning classifiers, tunes the top models, and selects the best model for default prediction.

## Workflow:
1. **Load Data:** Load the one-hot encoded dataset `model_dataset.parquet` and the selected features.
2. **Train/Test Split:** Split the data into an 80/20 train/test set, saving the test set for evaluation.
3. **Model Baseline Evaluation:** Perform 5-fold cross-validation on a subset of the training data for:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - LightGBM
   - CatBoost
4. **Hyperparameter Tuning:** Tune hyperparameters for the top boosting models (CatBoost, LightGBM, XGBoost) using randomized search.
5. **Model Selection:** Retrain the best performing model on the full training set and persist it.

> **Local version notes:** Re-written from Databricks PySpark to local scikit-learn/pandas. Includes memory optimization when loading Parquet files.'''))

cells.append(md("## Setup"))
cells.append(code(SETUP_SNIPPET))

cells.append(md("## 1. Load Parquet Dataset & Features"))
cells.append(code('''
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import gc
import json

# Load feature names
with open(MODELS_DIR / "final_features.json", "r") as f:
    final_features = json.load(f)
print(f"Loaded {len(final_features)} features.")

# Load Parquet memory-efficiently
def load_parquet_memory_efficient(file_path):
    schema = pq.read_schema(file_path)
    cols = schema.names
    dfs = []
    chunk_size = 25
    for i in range(0, len(cols), chunk_size):
        chunk_cols = cols[i:i+chunk_size]
        df_chunk = pd.read_parquet(file_path, columns=chunk_cols)
        for col in df_chunk.columns:
            if df_chunk[col].dtype == 'float64':
                df_chunk[col] = df_chunk[col].astype('float32')
            elif df_chunk[col].dtype == 'int64':
                min_val, max_val = df_chunk[col].min(), df_chunk[col].max()
                if min_val >= 0:
                    if max_val < 255:
                        df_chunk[col] = df_chunk[col].astype('uint8')
                    elif max_val < 65535:
                        df_chunk[col] = df_chunk[col].astype('uint16')
                    else:
                        df_chunk[col] = df_chunk[col].astype('uint32')
                else:
                    if min_val > -128 and max_val < 127:
                        df_chunk[col] = df_chunk[col].astype('int8')
                    elif min_val > -32768 and max_val < 32767:
                        df_chunk[col] = df_chunk[col].astype('int16')
                    else:
                        df_chunk[col] = df_chunk[col].astype('int32')
            elif df_chunk[col].dtype == 'object':
                df_chunk[col] = df_chunk[col].astype('category')
        dfs.append(df_chunk)
        gc.collect()
    df = pd.concat(dfs, axis=1)
    return df

df = load_parquet_memory_efficient(PROCESSED_DIR / "model_dataset.parquet")
print(f"Loaded dataset: {df.shape}")
'''))

cells.append(md("## 2. Train/Test Split & Sampling"))
cells.append(code('''
from sklearn.model_selection import train_test_split
import pickle

X = df[final_features]
y = df["TARGET"]

# 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Save test split for evaluation and explainability (ignored in git)
with open(MODELS_DIR / "X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open(MODELS_DIR / "y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
print("Saved test sets to models/")

# Create a 25% sample of training set for fast hyperparameter tuning and CV
X_train_cv, _, y_train_cv, _ = train_test_split(
    X_train, y_train, train_size=0.25, stratify=y_train, random_state=42
)
print(f"Tuning CV sample shape: {X_train_cv.shape}")

# Scale pos weight for XGBoost class balancing
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"Imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")

del df, X, y
gc.collect()
'''))

cells.append(md("## 3. Baseline Model Comparison (5-Fold CV)"))
cells.append(code('''
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(model, name):
    scores = cross_val_score(model, X_train_cv, y_train_cv, cv=skf, scoring="roc_auc", n_jobs=1)
    return {
        "Model": name,
        "ROC_AUC_Mean": scores.mean(),
        "ROC_AUC_STD": scores.std()
    }

# 1. Logistic Regression Pipeline
log_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])
log_results = evaluate_model(log_model, "Logistic Regression")
print(log_results)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1)
rf_results = evaluate_model(rf_model, "Random Forest")
print(rf_results)

# 3. XGBoost
xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss")
xgb_results = evaluate_model(xgb_model, "XGBoost")
print(xgb_results)

# 4. LightGBM
lgb_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
lgb_results = evaluate_model(lgb_model, "LightGBM")
print(lgb_results)

# 5. CatBoost
cat_model = CatBoostClassifier(iterations=200, learning_rate=0.05, verbose=0, random_state=42)
cat_results = evaluate_model(cat_model, "CatBoost")
print(cat_results)

# Combine results
results_df = pd.DataFrame([
    log_results,
    rf_results,
    xgb_results,
    lgb_results,
    cat_results
]).sort_values("ROC_AUC_Mean", ascending=False)
display(results_df)
'''))

cells.append(md("## 4. Hyperparameter Tuning"))
cells.append(code('''
from sklearn.model_selection import RandomizedSearchCV

# 1. Tune CatBoost
cat_grid = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "iterations": [200, 400],
    "l2_leaf_reg": [1, 3, 5]
}
cat_search = RandomizedSearchCV(
    estimator=CatBoostClassifier(verbose=0, random_state=42),
    param_distributions=cat_grid,
    n_iter=6,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=1
)
cat_search.fit(X_train_cv, y_train_cv)
best_cat = cat_search.best_estimator_
print("CatBoost Tuned AUC:", cat_search.best_score_)

# 2. Tune LightGBM
lgb_grid = {
    "num_leaves": [31, 63],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400],
    "max_depth": [6, 8, -1]
}
lgb_search = RandomizedSearchCV(
    estimator=LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
    param_distributions=lgb_grid,
    n_iter=6,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=1
)
lgb_search.fit(X_train_cv, y_train_cv)
best_lgb = lgb_search.best_estimator_
print("LightGBM Tuned AUC:", lgb_search.best_score_)

# 3. Tune XGBoost
xgb_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 400]
}
xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42),
    param_distributions=xgb_grid,
    n_iter=6,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=1
)
xgb_search.fit(X_train_cv, y_train_cv)
best_xgb = xgb_search.best_estimator_
print("XGBoost Tuned AUC:", xgb_search.best_score_)
'''))

cells.append(md("## 5. Tuned Model Comparison & Dynamic Selection"))
cells.append(code('''
# Evaluate tuned models
tuned_results = []
for name, model in [
    ("CatBoost", best_cat),
    ("LightGBM", best_lgb),
    ("XGBoost", best_xgb)
]:
    scores = cross_val_score(model, X_train_cv, y_train_cv, cv=skf, scoring="roc_auc", n_jobs=1)
    tuned_results.append({
        "Model": name,
        "ROC_AUC": scores.mean(),
        "STD": scores.std()
    })

tuned_df = pd.DataFrame(tuned_results).sort_values("ROC_AUC", ascending=False)
display(tuned_df)

# Select the best model dynamically
best_model_name = tuned_df.iloc[0]["Model"]
best_model = {
    "CatBoost": best_cat,
    "LightGBM": best_lgb,
    "XGBoost": best_xgb
}[best_model_name]

print(f"Dynamically selected the best model: {best_model_name}")

# Retrain on the full training set
print("Fitting best model on the complete training set...")
best_model.fit(X_train, y_train)
'''))

cells.append(md("## 6. Persist Best Model and Results"))
cells.append(code('''
import joblib

# Save best model pickle
model_filename = MODELS_DIR / f"home_credit_{best_model_name.lower()}.pkl"
joblib.dump(best_model, model_filename)
print(f"Saved best model: {model_filename}")

# Save tuned comparisons
tuned_df.to_csv(MODELS_DIR / "tuned_results.csv", index=False)
print("Saved tuned results comparison.")
'''))

save(cells, "03_ModelTraining.ipynb", "03")
