import os
import sys
import pickle
import joblib
import json
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split

BASE_DIR = Path("D:/Downloads/credit_risk")
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def one_hot_encode_and_align(df, categorical_cols, final_features):
    df_encoded = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], dummy_na=True)
    df_encoded = df_encoded.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', str(x)))
    
    # Ensure all final features are present
    for col in final_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    return df_encoded[final_features], df_encoded["TARGET"]

def main():
    print("Loading test data...")
    with open(MODELS_DIR / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(MODELS_DIR / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
        
    print("Loading CatBoost model...")
    best_model = joblib.load(MODELS_DIR / "home_credit_catboost.pkl")
    
    print("Computing baseline predictions...")
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, 
        y_prob, 
        n_bins=10,
        strategy='uniform'
    )
    
    avg_gap_before = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    print(f"Before calibration - average gap: {avg_gap_before:.6f}")
    
    print("Loading feature metadata...")
    with open(MODELS_DIR / "final_features.json", "r") as f:
        final_features = json.load(f)
        
    import pyarrow.parquet as pq
    schema = pq.read_schema(PROCESSED_DIR / "model_dataset_v2.parquet")
    parquet_cols = set(schema.names)
    
    categorical_cols = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
        'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    
    cols_to_load = ["TARGET", "SK_ID_CURR"]
    for col in final_features:
        if col in parquet_cols and col not in cols_to_load:
            cols_to_load.append(col)
    for col in categorical_cols:
        if col in parquet_cols and col not in cols_to_load:
            cols_to_load.append(col)
            
    print("Loading skeleton of dataset to identify split indices...")
    df_skeleton = pd.read_parquet(PROCESSED_DIR / "model_dataset_v2.parquet", columns=["TARGET", "SK_ID_CURR"])
    
    # Recreate training set indices
    train_idx = df_skeleton.index.difference(X_test.index)
    
    print("Splitting training data for calibration...")
    _, calib_idx = train_test_split(
        train_idx, 
        test_size=0.2, 
        random_state=42,
        stratify=df_skeleton.loc[train_idx, "TARGET"]
    )
    
    calib_sk_ids = df_skeleton.loc[calib_idx, "SK_ID_CURR"].tolist()
    
    # Free memory
    del df_skeleton
    import gc
    gc.collect()
    
    print(f"Loading columns from parquet for calibration sample ({len(calib_sk_ids)} rows)...")
    df_calib = pd.read_parquet(
        PROCESSED_DIR / "model_dataset_v2.parquet", 
        columns=cols_to_load,
        filters=[("SK_ID_CURR", "in", calib_sk_ids)]
    )
    
    # Downcast floats
    for col in df_calib.columns:
        if df_calib[col].dtype == 'float64':
            df_calib[col] = df_calib[col].astype('float32')
                
    print("One-hot encoding and aligning calibration split...")
    X_calib_sample, y_calib_sample = one_hot_encode_and_align(df_calib, categorical_cols, final_features)
    
    # Free memory
    del df_calib
    gc.collect()
    
    print("Fitting CalibratedClassifierCV on prefitted best_model (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(
        best_model,
        cv='prefit',
        method='sigmoid'
    )
    calibrated_model.fit(X_calib_sample, y_calib_sample)
    
    # Predict calibrated probabilities
    y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    
    fraction_of_positives_cal, mean_predicted_value_cal = calibration_curve(
        y_test,
        y_prob_calibrated,
        n_bins=10,
        strategy='uniform'
    )
    
    avg_gap_after = np.mean(np.abs(fraction_of_positives_cal - mean_predicted_value_cal))
    print(f"After calibration - average gap: {avg_gap_after:.6f}")
    
    brier_before = float(np.mean((y_prob - y_test) ** 2))
    brier_after = float(np.mean((y_prob_calibrated - y_test) ** 2))
    print(f"Brier score - Before: {brier_before:.6f}, After: {brier_after:.6f}")
    
    # Save the calibration results JSON
    calibration_data = {
        "before": {
            "mean_predicted": mean_predicted_value.tolist(),
            "actual_default_rate": fraction_of_positives.tolist()
        },
        "after": {
            "mean_predicted": mean_predicted_value_cal.tolist(),
            "actual_default_rate": fraction_of_positives_cal.tolist()
        },
        "brier_score_before": brier_before,
        "brier_score_after": brier_after
    }
    
    output_path = BASE_DIR / "static" / "calibration_data.json"
    with open(output_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"Saved calibration data to {output_path}")

    # Also save the calibrated model
    joblib.dump(calibrated_model, MODELS_DIR / "home_credit_catboost_calibrated.pkl")
    print(f"Saved calibrated model to {MODELS_DIR / 'home_credit_catboost_calibrated.pkl'}")

if __name__ == "__main__":
    main()
