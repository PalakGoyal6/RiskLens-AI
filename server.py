import os
import pickle
import joblib
import pandas as pd
import numpy as np
import shap
import io
import sqlite3
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
import auth

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RiskLens AI - SHAP Explainability Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

import traceback
from datetime import datetime
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def validation_exception_handler(request: Request, exc: Exception):
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"Exception occurred at {datetime.now().isoformat()}:\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
        f.write("\n" + "="*80 + "\n")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "detail": str(exc)},
    )

from fastapi.security import OAuth2PasswordBearer
from fastapi import status

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = auth.decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload

FEATURE_DISPLAY_NAMES = {
    "EXT_SOURCE_2": "Credit Bureau Score A",
    "EXT_SOURCE_3": "Credit Bureau Score B",
    "EXT_SOURCE_1": "Credit Bureau Score C",
    "DAYS_EMPLOYED": "Employment Duration",
    "AMT_GOODS_PRICE": "Purchase Price",
    "BUREAU_DEBT_CREDIT_RATIO": "Debt-to-Credit Ratio",
    "PREV_CREDIT_APP_RATIO": "Credit-to-Application Ratio",
    "POS_COUNT": "No. of Previous POS Loans",
    "RECENT_LATE_RATIO": "Recent Late Payment Rate",
    "CREDIT_TERM": "Credit Term Duration",
    "AMT_ANNUITY": "Annual Loan Annuity",
    "AMT_CREDIT": "Total Credit Amount",
    "LATE_PAYMENT_RATIO": "Late Payment History Ratio",
    "CODE_GENDER_M": "Gender: Male",
    "CODE_GENDER_F": "Gender: Female",
    "NAME_FAMILY_STATUS_Married": "Family Status: Married",
    "FLAG_DOCUMENT_3": "Document 3 Submitted",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "CNT_CHILDREN": "Number of Children",
    "OWN_CAR_AGE": "Age of Applicant's Car",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio",
    "CC_UTILIZATION_TREND_6M": "Credit Card Utilization Trend (6M)",
    "CC_MAX_UTILIZATION": "Credit Card Max Utilization",
    "CC_AVG_UTILIZATION": "Credit Card Avg Utilization",
    "CC_OVER_LIMIT_RATIO": "Credit Card Over-Limit Ratio",
    "CC_DRAWINGS_DIFF_6M": "Credit Card Draw Difference (6M)",
    "PREV_REFUSED_RATIO_RECENT3": "Recent Prev Loan Refusal Rate (Last 3)",
    "PREV_REFUSED_RATIO": "Previous Loan Refusal Rate",
    "PREV_REFUSED_COUNT": "Previous Refused Loan Count",
    "PREV_DAYS_SINCE_LAST_REFUSED": "Days Since Last Loan Refusal",
    "BUREAU_LOAN_COUNT": "Credit Bureau Active Loan Count",
    "BUREAU_ACTIVE_COUNT": "Credit Bureau Active Loans",
    "BUREAU_ACTIVE_CLOSED_RATIO": "Credit Bureau Active-to-Closed Ratio",
    "BUREAU_AVG_DPD_RATIO": "Credit Bureau Average DPD Ratio",
    "BUREAU_DPD_TREND": "Credit Bureau Delinquency Trend",
    "BUREAU_AVG_DEBT": "Credit Bureau Average Debt",
    "BUREAU_AVG_CREDIT": "Credit Bureau Average Credit Limit",
    "INST_LATE_RATIO_3M": "Installment Late Rate (3M)",
    "INST_LATE_RATIO_6M": "Installment Late Rate (6M)",
    "INST_LATE_RATIO_12M": "Installment Late Rate (12M)",
    "MAX_PAYMENT_DELAY": "Max Installment Delay (Days)",
    "AVG_PAYMENT_DELAY": "Average Installment Delay (Days)",
    "RECENT_AVG_DELAY": "Recent Average Payment Delay",
    "UNDERPAID_RATIO": "Underpayment Ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-Income Ratio",
    "POS_DPD_RATIO": "POS DPD History Ratio",
    "POS_DPD_TREND": "POS Overdue Payment Trend",
    "POS_DPD_SLOPE": "POS Overdue Slope",
    "POS_COMPLETION_RATE_DIFF": "POS Completion Rate Difference"
}

FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_2": "Normalized score from external credit bureau (0 = high risk, 1 = low risk). This is the single strongest predictor of default in this model.",
    "EXT_SOURCE_3": "Normalized score from external credit bureau (0 = high risk, 1 = low risk). This is the second strongest predictor of default in this model.",
    "EXT_SOURCE_1": "Normalized score from external credit bureau (0 = high risk, 1 = low risk). Highly predictive of default risk.",
    "DAYS_EMPLOYED": "Number of days the applicant has been employed. Shorter employment durations typically indicate higher risk.",
    "AMT_GOODS_PRICE": "The price of the goods for which the loan is given. High values increase the total borrowing amount.",
    "BUREAU_DEBT_CREDIT_RATIO": "Total outstanding debt relative to total active credit limit from the credit bureau. Higher ratio indicates higher risk.",
    "PREV_CREDIT_APP_RATIO": "Ratio of credit amount approved compared to what was initially requested. Lower ratios suggest lending constraints.",
    "POS_COUNT": "Number of previous Point-of-Sale cash loans. High frequency of short-term loans can indicate financial strain.",
    "RECENT_LATE_RATIO": "The percentage of payments delayed in the last few installments. Directly indicates delinquency risk.",
    "CREDIT_TERM": "Estimated duration of the loan term. Longer terms increase exposure to default risk.",
    "AMT_ANNUITY": "Annual loan installment payment. Higher annuities increase the monthly repayment burden on the applicant.",
    "AMT_CREDIT": "Total credit amount requested by the applicant (also known as total loan amount). Larger credit sizes increase exposure to risk.",
    "LATE_PAYMENT_RATIO": "Overall history of late payments on previous loans. High ratio shows consistent repayment delays.",
    "CODE_GENDER_M": "Indicates if the applicant is male. Historically correlates with slightly higher default rates in this dataset.",
    "CODE_GENDER_F": "Indicates if the applicant is female. Historically correlates with lower default rates in this dataset.",
    "NAME_FAMILY_STATUS_Married": "Indicates if the applicant is married. Married applicants tend to show slightly higher stability and lower risk.",
    "FLAG_DOCUMENT_3": "Indicates if the main loan application document (Document 3) was submitted. Missing documentation increases risk flags.",
    "AMT_INCOME_TOTAL": "The total annual income reported by the applicant. Higher income typically lowers default risk.",
    "CNT_CHILDREN": "The number of children in the applicant's household. More dependents can increase financial commitments.",
    "OWN_CAR_AGE": "The age of the applicant's primary car. Older cars can indicate lower assets or higher maintenance costs.",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio. Measures the credit size relative to total annual income.",
    "CC_UTILIZATION_TREND_6M": "The 6-month trend of the credit card balance utilization rate. An upward trend suggests growing reliance on credit lines and higher risk.",
    "CC_MAX_UTILIZATION": "The maximum credit card balance utilization rate observed historically across all statement periods.",
    "CC_AVG_UTILIZATION": "The average credit card balance utilization rate observed historically across all statement periods.",
    "CC_OVER_LIMIT_RATIO": "The frequency of months where the credit card balance exceeded the approved credit limit.",
    "CC_DRAWINGS_DIFF_6M": "The difference in average monthly credit card cash/purchase drawings between the last 6 months and the historical average.",
    "PREV_REFUSED_RATIO_RECENT3": "The percentage of the last 3 previous loan applications that were refused by the bank.",
    "PREV_REFUSED_RATIO": "The overall percentage of previous loan applications that were refused by the bank.",
    "PREV_REFUSED_COUNT": "The total count of previous applications that were refused.",
    "PREV_DAYS_SINCE_LAST_REFUSED": "Number of days elapsed since the client's last loan application refusal.",
    "BUREAU_LOAN_COUNT": "The total number of past loans the client has registered in the external credit bureau.",
    "BUREAU_ACTIVE_COUNT": "The number of currently active loans the client has with other financial institutions.",
    "BUREAU_ACTIVE_CLOSED_RATIO": "The ratio of active loans to closed loans in the credit bureau records.",
    "BUREAU_AVG_DPD_RATIO": "The percentage of months where the bureau reports the client had payments overdue.",
    "BUREAU_DPD_TREND": "The trend/slope of overdue days reported by the credit bureau. Positive slope suggests worsening delinquency.",
    "BUREAU_AVG_DEBT": "The average outstanding debt amount across all bureau-reported credit accounts.",
    "BUREAU_AVG_CREDIT": "The average credit limit of accounts reported in the credit bureau.",
    "INST_LATE_RATIO_3M": "The percentage of loan installments paid late in the last 3 months.",
    "INST_LATE_RATIO_6M": "The percentage of loan installments paid late in the last 6 months.",
    "INST_LATE_RATIO_12M": "The percentage of loan installments paid late in the last 12 months.",
    "MAX_PAYMENT_DELAY": "The maximum number of days a payment was delayed past the due date.",
    "AVG_PAYMENT_DELAY": "The average delay in days across all historical installment payments.",
    "RECENT_AVG_DELAY": "The average delay in days of installment payments in recent months.",
    "UNDERPAID_RATIO": "The ratio of installments paid below the billed amount.",
    "ANNUITY_INCOME_RATIO": "The monthly loan annuity payment divided by the applicant's monthly income.",
    "POS_DPD_RATIO": "The percentage of monthly points-of-sale cash loan records with overdue payments.",
    "POS_DPD_TREND": "The trend in days past due for POS cash accounts. An upward trend indicates worsening risk.",
    "POS_DPD_SLOPE": "The rate of change of days past due over time for Point-of-Sale cash accounts.",
    "POS_COMPLETION_RATE_DIFF": "The difference between the expected and actual completion rate of previous POS loans."
}

def build_feature_mapping_text(display_names: dict, descriptions: dict) -> str:
    lines = ["Feature Name Reference (always use Plain English Name when discussing risk factors):"]
    for raw_name, plain_name in display_names.items():
        description = descriptions.get(raw_name, "")
        lines.append(f"- {raw_name} -> \"{plain_name}\" ({description})")
    return "\n".join(lines)

FEATURE_MAPPING_TEXT = build_feature_mapping_text(FEATURE_DISPLAY_NAMES, FEATURE_DESCRIPTIONS)

def get_gemini_narration(probability: float, factors_text: str) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not found. Skipping real Gemini API call.")
        return None
        
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""You are assisting a bank loan officer.
Write exactly 2 sentences maximum. First sentence: state the predicted probability and risk category. Second sentence: name the top 3 risk factors only, in plain English. Do not list more than 3 factors. Do not use parenthetical values.
Strict Constraint: Do NOT make a final approve/decline recommendation. The decision must stay with the human officer; your job is strictly to interpret the factors.

Applicant Default Probability: {probability:.1%}
Key SHAP Risk Factors (Direction & Magnitude):
{factors_text}

Provide only the 2-sentence explanation as text. No markdown, no introductions, no recommendations."""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="Write exactly 2 sentences maximum. First sentence: state the predicted probability and risk category. Second sentence: name the top 3 risk factors only, in plain English. Do not list more than 3 factors. Do not use parenthetical values.",
                temperature=0.1,
                max_output_tokens=150,
            )
        )
        return response.text.strip()
    except Exception as e:
        print("Gemini API call failed:", e)
        return None

def get_openai_narration(probability: float, factors_text: str) -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        return None
        
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    
    try:
        import requests
        prompt = f"""You are assisting a bank loan officer.
Write exactly 2 sentences maximum. First sentence: state the predicted probability and risk category. Second sentence: name the top 3 risk factors only, in plain English. Do not list more than 3 factors. Do not use parenthetical values.
Strict Constraint: Do NOT make a final approve/decline recommendation. The decision must stay with the human officer; your job is strictly to interpret the factors.

Applicant Default Probability: {probability:.1%}
Key SHAP Risk Factors (Direction & Magnitude):
{factors_text}

Provide only the 2-sentence explanation as text. No markdown, no introductions, no recommendations."""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a factual assistant explaining credit risk scoring factors without making approve/decline judgments. Write exactly 2 sentences maximum. First sentence: state the predicted probability and risk category. Second sentence: name the top 3 risk factors only, in plain English. Do not list more than 3 factors. Do not use parenthetical values."},
                {"role": "user", "content": prompt}
            ]
        }
        if "gpt-5" in model.lower() or "o1" in model.lower() or "o3" in model.lower():
            payload["max_completion_tokens"] = 150
        else:
            payload["max_tokens"] = 150
            payload["temperature"] = 0.1
        
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()
        else:
            print(f"OpenAI call failed (status {response.status_code}): {response.text}")
            return None
    except Exception as e:
        print("OpenAI call error:", e)
        return None

def generate_fallback_narration(probability: float, explanation: list) -> str:
    # risk category
    if probability >= 0.25:
        risk_cat = "High Risk"
    elif probability >= 0.10:
        risk_cat = "Moderate Risk"
    else:
        risk_cat = "Low Risk"
        
    # First sentence
    sent1 = f"This applicant has a {probability:.1%} predicted default probability, placing them in the {risk_cat} category."
    
    # Second sentence: name the top 3 risk factors only with low/high prefixes
    formatted_factors = []
    ext_sources = []
    other_factors = []
    
    for f in explanation[:3]:
        feat_name = f["feature"]
        shap_val = f["shap_value"]
        
        display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace("_", " ").title())
        
        if feat_name in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            letter = "A" if feat_name == "EXT_SOURCE_2" else ("B" if feat_name == "EXT_SOURCE_3" else "C")
            ext_sources.append((letter, shap_val))
        else:
            prefix = ""
            if "RATIO" in feat_name or "RATE" in feat_name or "DELAY" in feat_name or "AMT" in feat_name or "COUNT" in feat_name or "AGE" in feat_name:
                prefix = "high " if shap_val > 0 else "low "
            elif "DAYS_EMPLOYED" in feat_name:
                prefix = "short " if shap_val > 0 else "stable "
            elif "Married" in feat_name:
                prefix = "unfavorable " if shap_val > 0 else "favorable "
            else:
                prefix = "high " if shap_val > 0 else "low "
            
            other_factors.append(f"a {prefix}{display_name}")
            
    if ext_sources:
        ext_sources.sort(key=lambda x: x[0])
        letters = [x[0] for x in ext_sources]
        shap_vals = [x[1] for x in ext_sources]
        
        avg_shap = sum(shap_vals) / len(shap_vals)
        prefix = "low " if avg_shap > 0 else "high "
        
        if len(letters) == 1:
            formatted_factors.append(f"{prefix}Credit Bureau Score {letters[0]}")
        elif len(letters) == 2:
            formatted_factors.append(f"{prefix}Credit Bureau Scores {letters[0]} and {letters[1]}")
        else:
            formatted_factors.append(f"{prefix}Credit Bureau Scores A, B and C")
            
    for fact in other_factors:
        formatted_factors.append(fact)
        
    if not formatted_factors:
        sent2 = "There are no significant risk factors identified."
    elif len(formatted_factors) == 1:
        sent2 = f"The primary driver is {formatted_factors[0]}."
    elif len(formatted_factors) == 2:
        sent2 = f"The primary drivers are {formatted_factors[0]}, and {formatted_factors[1]}."
    else:
        sent2 = f"The primary drivers are {formatted_factors[0]}, {formatted_factors[1]}, and {formatted_factors[2]}."
        
    sent2 = sent2.replace(".,", ".").replace("and a", "and a").replace(",,", ",")
    return f"{sent1} {sent2}"

def generate_narration(probability: float, explanation: list) -> str:
    # 1. Format factors for the prompt
    top_factors = explanation[:6]
    factors_lines = []
    for f in top_factors:
        feat_name = f["feature"]
        raw_val = f["feature_value"]
        shap_val = f["shap_value"]
        
        display_name = FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace("_", " ").title())
        
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            val_str = "N/A"
        elif "RATIO" in feat_name or "RATE" in feat_name:
            val_str = f"{raw_val * 100:.1f}%"
        elif "DAYS_EMPLOYED" in feat_name:
            if raw_val < 0:
                val_str = f"{abs(int(raw_val))} days of employment"
            else:
                val_str = "unemployed"
        elif "PRICE" in feat_name or "CREDIT" in feat_name or "ANNUITY" in feat_name or "INCOME" in feat_name:
            val_str = f"${raw_val:,.2f}"
        else:
            val_str = f"{raw_val:.2f}" if isinstance(raw_val, float) else str(raw_val)
            
        direction = "increases risk" if shap_val > 0 else "decreases risk"
        factors_lines.append(f"- {display_name} = {val_str} ({direction} by {abs(shap_val):.4f})")
        
    factors_text = "\n".join(factors_lines)
    
    # 2. Try calling Gemini
    narration = get_gemini_narration(probability, factors_text)
    
    # 3. Try calling OpenAI (for GPT-5-mini / custom proxy)
    if not narration:
        narration = get_openai_narration(probability, factors_text)
        
    # 4. Fallback if both missing or call fails
    if not narration:
        narration = generate_fallback_narration(probability, explanation)
        
    return narration

def log_prediction(applicant_id: int, officer_id: Optional[int], probability: float, shap_values_dict: dict, narration: Optional[str] = None) -> int:
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        if probability >= 0.25:
            risk_label = "high"
        elif probability >= 0.10:
            risk_label = "medium"
        else:
            risk_label = "low"
            
        shap_json = json.dumps(shap_values_dict)
        
        cursor.execute("""
            INSERT INTO predictions (applicant_id, officer_id, predicted_probability, risk_label, shap_values, narration)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (applicant_id, officer_id, probability, risk_label, shap_json, narration))
        prediction_id = cursor.lastrowid
        conn.commit()
        return prediction_id
    finally:
        conn.close()

class LoginRequest(BaseModel):
    username: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class DecisionRequest(BaseModel):
    prediction_id: int
    decision: str  # approved, declined, escalated
    notes: Optional[str] = None

class RecalculateRequest(BaseModel):
    features: Dict[str, Any]

# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Global variables to store loaded models and cached SHAP data
model = None
shap_cache = None
explainer = None
X_test = None
X_test_mean = None

@app.on_event("startup")
def load_assets():
    global model, shap_cache, explainer, X_test, X_test_mean
    print("Loading models and cached assets...")
    
    # Load CatBoost model
    model_path = MODELS_DIR / "home_credit_catboost.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = joblib.load(model_path)
    
    # Load SHAP cache
    cache_path = MODELS_DIR / "shap_cache.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(f"SHAP cache not found at {cache_path}")
    with open(cache_path, "rb") as f:
        shap_cache = pickle.load(f)
        
    # Load original X_test for full what-if support
    X_test_path = MODELS_DIR / "X_test.pkl"
    with open(X_test_path, "rb") as f:
        X_test = pickle.load(f)
        
    # Precalculate baseline mean feature values
    X_test_mean = X_test.mean()
        
    # Initialize Explainer
    explainer = shap.TreeExplainer(model)
    
    # Database migration: add narration column to predictions table if not exists
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [row[1] for row in cursor.fetchall()]
        if "narration" not in columns:
            cursor.execute("ALTER TABLE predictions ADD COLUMN narration TEXT")
            conn.commit()
            print("Database migration: Added narration column to predictions table.")
    except Exception as e:
        print("Database migration check failed:", e)
    finally:
        conn.close()
        
    print("Assets loaded successfully.")

# Pydantic models for request bodies
class WhatIfRequest(BaseModel):
    applicant_id: int
    updates: Dict[str, Optional[float]]

class PredictRequest(BaseModel):
    features: Dict[str, Any]

class ExplainRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/")
def read_root():
    landing_file = STATIC_DIR / "landing.html"
    if not landing_file.exists():
        landing_file = STATIC_DIR / "index.html"
    return FileResponse(landing_file)

@app.get("/login")
def read_login():
    login_file = STATIC_DIR / "login.html"
    if not login_file.exists():
        login_file = STATIC_DIR / "index.html"
    return FileResponse(login_file)

@app.get("/dashboard")
def read_dashboard():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(status_code=404, content={"message": "Frontend not built yet. static/index.html is missing."})
    return FileResponse(index_file)

@app.post("/api/auth/demo")
@app.get("/api/auth/demo")
def demo_login():
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, password_hash, role FROM users WHERE username = 'officer1'")
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="Demo user 'officer1' not found in database.")
        user_id, password_hash, role = row
        token_data = {
            "id": user_id,
            "username": "officer1",
            "role": role
        }
        token = auth.create_access_token(data=token_data)
        return {
            "token": token,
            "access_token": token,
            "token_type": "bearer",
            "user": token_data
        }
    finally:
        conn.close()

@app.get("/api/dashboard/home")
def get_dashboard_home(user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        # 1. Fetch recent activity (last 5 decisions)
        cursor.execute("""
            SELECT d.id as decision_id, d.decision, d.notes, d.timestamp as decision_time,
                   p.applicant_id, p.predicted_probability,
                   a.name as applicant_name, u.username as officer_name
            FROM decisions d
            JOIN predictions p ON d.prediction_id = p.id
            JOIN applicants a ON p.applicant_id = a.applicant_id
            LEFT JOIN users u ON d.officer_id = u.id
            ORDER BY d.timestamp DESC
            LIMIT 5
        """)
        recent_rows = cursor.fetchall()
        recent_activity = [dict(r) for r in recent_rows]
        
        # 2. Get all IDs that have decisions to filter them out of pending cases
        cursor.execute("SELECT DISTINCT p.applicant_id FROM predictions p JOIN decisions d ON d.prediction_id = p.id")
        decided_ids = {r[0] for r in cursor.fetchall()}
        
        # 3. Get pending cases from sample cohort
        ids = shap_cache["ids"]
        sample_df = X_test.loc[ids]
        probs = model.predict_proba(sample_df)[:, 1]
        
        pending_cases = []
        for i, app_id in enumerate(ids):
            if int(app_id) not in decided_ids:
                # Retrieve applicant name from database
                cursor.execute("SELECT name FROM applicants WHERE applicant_id = ?", (int(app_id),))
                name_row = cursor.fetchone()
                name = name_row["name"] if name_row else "Unknown Applicant"
                
                pending_cases.append({
                    "id": int(app_id),
                    "name": name,
                    "probability": float(probs[i]),
                    "risk_classification": "High Risk" if probs[i] >= 0.25 else ("Moderate Risk" if probs[i] >= 0.10 else "Low Risk")
                })
                if len(pending_cases) >= 5:
                    break
        
        # 4. Gather quick stats
        cursor.execute("SELECT COUNT(*) FROM applicants")
        total_applicants = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'approved'")
        total_approved = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM decisions WHERE decision = 'declined'")
        total_declined = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM decisions")
        total_decisions = cursor.fetchone()[0]
        
        # Pending cases in the sample cohort
        all_pending_in_cohort = []
        for i, app_id in enumerate(ids):
            if int(app_id) not in decided_ids:
                all_pending_in_cohort.append(float(probs[i]))
        
        avg_risk_pending = sum(all_pending_in_cohort) / len(all_pending_in_cohort) if all_pending_in_cohort else 0.0
        
        stats = {
            "total_applicants": total_applicants,
            "total_decisions": total_decisions,
            "total_approved": total_approved,
            "total_rejected": total_declined,
            "total_pending": len(all_pending_in_cohort),
            "avg_risk_pending": avg_risk_pending
        }
        
        return {
            "recent_activity": recent_activity,
            "pending_cases": pending_cases,
            "stats": stats
        }
    finally:
        conn.close()

@app.get("/api/applicants/pending")
def get_pending_applicants(user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        # Get decided IDs
        cursor.execute("SELECT DISTINCT p.applicant_id FROM predictions p JOIN decisions d ON d.prediction_id = p.id")
        decided_ids = {r[0] for r in cursor.fetchall()}
        
        ids = shap_cache["ids"]
        targets = shap_cache["targets"]
        sample_df = X_test.loc[ids]
        probs = model.predict_proba(sample_df)[:, 1]
        
        pending = []
        for i, app_id in enumerate(ids):
            if int(app_id) not in decided_ids:
                cursor.execute("SELECT name FROM applicants WHERE applicant_id = ?", (int(app_id),))
                name_row = cursor.fetchone()
                name = name_row["name"] if name_row else "Unknown Applicant"
                pending.append({
                    "id": int(app_id),
                    "name": name,
                    "target": int(targets[i]),
                    "probability": float(probs[i]),
                    "risk_group": "High Risk" if probs[i] >= 0.15 else "Safe"
                })
        # Sort pending by risk probability descending
        pending = sorted(pending, key=lambda x: x["probability"], reverse=True)
        return pending
    finally:
        conn.close()

@app.get("/api/summary")
def get_summary():
    """Returns general metrics and threshold details."""
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM applicants")
        db_count = cursor.fetchone()[0]
    except Exception:
        db_count = 61503
    finally:
        conn.close()
        
    return {
        "threshold": 0.15,
        "auc": 0.781,
        "precision": 0.266,
        "recall": 0.450,
        "f1": 0.334,
        "sample_size": len(shap_cache["ids"]),
        "total_applicants": db_count
    }

@app.get("/api/global-importance")
def get_global_importance():
    """Returns sorted global feature importance based on mean absolute SHAP values."""
    feature_names = shap_cache["feature_names"]
    shap_vals = shap_cache["shap_values"]
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    importance = []
    for name, imp in zip(feature_names, mean_abs_shap):
        importance.append({"feature": name, "importance": float(imp)})
        
    # Sort descending
    importance = sorted(importance, key=lambda x: x["importance"], reverse=True)
    return importance[:30]  # Top 30 features

@app.get("/api/beeswarm")
def get_beeswarm():
    """
    Returns data required for beeswarm plot:
    - For each feature in the top 15:
      - Array of points, where each point has:
        - shap_value: SHAP value for that instance
        - feature_value: Raw value of the feature for that instance
        - normalized_value: Value normalized between 0 and 1 (for color mapping)
        - applicant_id: ID of the applicant
    """
    feature_names = shap_cache["feature_names"]
    shap_vals = shap_cache["shap_values"]
    data = shap_cache["data"]
    ids = shap_cache["ids"]
    top_features = shap_cache["top_features"][:15]
    
    result = {}
    for feat_name in top_features:
        feat_idx = feature_names.index(feat_name)
        
        # Get raw data and SHAP values for this feature
        raw_vals = pd.to_numeric(data[:, feat_idx], errors='coerce')
        shaps = shap_vals[:, feat_idx]
        
        # Filter out NaN or handle division by zero
        non_nan_mask = ~np.isnan(raw_vals)
        if not np.any(non_nan_mask):
            continue
            
        feat_min = float(np.min(raw_vals[non_nan_mask]))
        feat_max = float(np.max(raw_vals[non_nan_mask]))
        feat_range = feat_max - feat_min if feat_max != feat_min else 1.0
        
        points = []
        # Downsample to 200 points to keep D3 browser force layout simulations extremely snappy!
        step = 5
        for idx in range(0, len(ids), step):
            app_id = ids[idx]
            val = raw_vals[idx]
            if np.isnan(val):
                normalized = 0.5  # default neutral color for NaNs
            else:
                normalized = (float(val) - feat_min) / feat_range
                
            points.append({
                "shap_value": float(shaps[idx]),
                "feature_value": float(val) if not np.isnan(val) else None,
                "normalized_value": normalized,
                "applicant_id": int(app_id)
            })
        
        result[feat_name] = points
        
    return result

@app.get("/api/applicants")
def get_applicants():
    """Returns a list of applicant summaries (ID, Target, predicted probability)."""
    ids = shap_cache["ids"]
    targets = shap_cache["targets"]
    
    # We will get predictions for all sample applicants
    sample_df = X_test.loc[ids]
    probs = model.predict_proba(sample_df)[:, 1]
    
    applicants = []
    for i, app_id in enumerate(ids):
        applicants.append({
            "id": int(app_id),
            "target": int(targets[i]),
            "probability": float(probs[i]),
            "risk_group": "High Risk" if probs[i] >= 0.15 else "Safe"
        })
        
    # Sort by probability descending to make default risk analysis easier
    applicants = sorted(applicants, key=lambda x: x["probability"], reverse=True)
    return applicants

@app.post("/api/what-if")
def calculate_what_if(payload: WhatIfRequest):
    """Calculates model prediction and SHAP values for a modified applicant vector."""
    app_id = payload.applicant_id
    updates = payload.updates
    
    if app_id not in X_test.index:
        raise HTTPException(status_code=404, detail="Applicant ID not found in test set.")
        
    # Copy the original row
    modified_row = X_test.loc[[app_id]].copy()
    
    # Apply updates with type safety (casting to original types)
    for col_name, value in updates.items():
        if value is None:
            continue
        if col_name in modified_row.columns:
            orig_type = X_test[col_name].dtype
            modified_row[col_name] = value
            modified_row[col_name] = modified_row[col_name].astype(orig_type)
            
            # Recalculate interaction terms if they depend on modified features!
            # Let's check which interactions are modified:
            # - CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL (if both present and updated)
            # Let's keep it simple: if updates don't recalculate automatically, we can do it:
            # But the user updates them via sliders directly! So they are independent inputs.
            
    # Calculate new probability
    prob = float(model.predict_proba(modified_row)[0, 1])
    
    # Calculate new SHAP explanation
    shap_explanation = explainer(modified_row)
    
    feature_names = list(X_test.columns)
    shap_values = shap_explanation.values[0]
    raw_values = modified_row.values[0]
    
    # Format list of features
    local_explanation = []
    for name, shap_val, raw_val in zip(feature_names, shap_values, raw_values):
        local_explanation.append({
            "feature": name,
            "shap_value": float(shap_val),
            "feature_value": float(raw_val) if not np.isnan(raw_val) else None
        })
        
    local_explanation = sorted(local_explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
    
    return {
        "applicant_id": app_id,
        "applicant_id": app_id,
        "probability": prob,
        "base_value": float(shap_explanation.base_values[0]),
        "explanation": local_explanation
    }

@app.post("/predict")
def predict_single(payload: PredictRequest):
    """
    Predicts credit default probability for a single applicant.
    Missing features are automatically imputed using baseline averages.
    """
    if model is None or X_test_mean is None or X_test is None:
        raise HTTPException(status_code=503, detail="Model or assets not loaded yet.")
        
    input_data = payload.features
    row_series = X_test_mean.copy()
    
    # Override defaults with inputs
    for col, val in input_data.items():
        if col in row_series.index:
            row_series[col] = val
            
    row_df = pd.DataFrame([row_series])
    for col in row_df.columns:
        row_df[col] = row_df[col].astype(X_test[col].dtype)
        
    prob = float(model.predict_proba(row_df)[0, 1])
    
    threshold = 0.15
    prediction = 1 if prob >= threshold else 0
    decision = "Rejected" if prob >= threshold else "Approved"
    
    return {
        "probability": prob,
        "prediction": prediction,
        "decision": decision
    }

@app.post("/explain")
def explain_single(payload: ExplainRequest):
    """
    Computes and returns model prediction and detailed SHAP contribution values for custom features.
    """
    if model is None or explainer is None or X_test_mean is None or X_test is None:
        raise HTTPException(status_code=503, detail="Model or assets not loaded yet.")
        
    input_data = payload.features
    row_series = X_test_mean.copy()
    
    for col, val in input_data.items():
        if col in row_series.index:
            row_series[col] = val
            
    row_df = pd.DataFrame([row_series])
    for col in row_df.columns:
        row_df[col] = row_df[col].astype(X_test[col].dtype)
        
    prob = float(model.predict_proba(row_df)[0, 1])
    shap_explanation = explainer(row_df)
    
    feature_names = list(row_df.columns)
    shap_values = shap_explanation.values[0]
    raw_values = row_df.values[0]
    
    explanation = []
    for name, shap_val, raw_val in zip(feature_names, shap_values, raw_values):
        explanation.append({
            "feature": name,
            "shap_value": float(shap_val),
            "feature_value": float(raw_val) if not np.isnan(raw_val) else None
        })
        
    explanation = sorted(explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
    
    return {
        "probability": prob,
        "base_value": float(shap_explanation.base_values[0]),
        "explanation": explanation
    }

@app.post("/batch")
async def batch_predict(file: UploadFile = File(...)):
    """
    Scores multiple applicants uploaded in a CSV file.
    Performs auto column alignment and baseline imputation for missing features.
    """
    if model is None or X_test_mean is None or X_test is None:
        raise HTTPException(status_code=503, detail="Model or assets not loaded yet.")
        
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
    if df.empty:
        return []
        
    # Check for ID columns
    id_col = None
    for col in df.columns:
        if col.lower() in ["sk_id_curr", "id", "applicant_id"]:
            id_col = col
            break
            
    # Process features: align columns with X_test
    cols_to_keep = [col for col in df.columns if col in X_test.columns]
    aligned_df = df[cols_to_keep].copy()
    
    # Broadcast baseline values for missing features
    for col in X_test.columns:
        if col not in aligned_df.columns:
            aligned_df[col] = X_test_mean[col]
            
    # Reorder columns
    aligned_df = aligned_df[X_test.columns]
    
    # Fill remaining NaNs from input file
    aligned_df = aligned_df.fillna(X_test_mean)
    
    # Cast column types
    for col in aligned_df.columns:
        aligned_df[col] = aligned_df[col].astype(X_test[col].dtype)
        
    # Predict probabilities
    probs = model.predict_proba(aligned_df)[:, 1]
    
    threshold = 0.15
    results = []
    for i, prob in enumerate(probs):
        app_id = int(df.iloc[i][id_col]) if id_col else i
        prediction = 1 if prob >= threshold else 0
        decision = "Rejected" if prob >= threshold else "Approved"
        
        results.append({
            "id": app_id,
            "probability": float(prob),
            "prediction": prediction,
            "decision": decision
        })
        
    return results

@app.post("/auth/login")
@app.post("/api/auth/login")
def login(payload: LoginRequest):
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        username_clean = payload.username.strip() if payload.username else ""
        cursor.execute("SELECT id, password_hash, role FROM users WHERE username = ?", (username_clean,))
        row = cursor.fetchone()
        if not row:
            print(f"Login failed: User '{username_clean}' not found in database.")
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        
        user_id, password_hash, role = row
        if not auth.verify_password(payload.password, password_hash):
            print(f"Login failed: Incorrect password provided for user '{username_clean}'.")
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        
        # Create JWT token
        token_data = {
            "id": user_id,
            "username": username_clean,
            "role": role
        }
        token = auth.create_access_token(data=token_data)
        print(f"Login success: User '{username_clean}' authenticated successfully. Role: {role}")
        return {
            "token": token,
            "access_token": token,
            "token_type": "bearer",
            "user": token_data
        }
    finally:
        conn.close()

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "loan_officer"

@app.post("/auth/register")
def register(payload: RegisterRequest, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only administrators can register new officer accounts.")
        
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (payload.username.strip(),))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already exists.")
            
        hashed = auth.hash_password(payload.password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (payload.username.strip(), hashed, payload.role)
        )
        conn.commit()
        return {"success": True, "message": f"User {payload.username} registered successfully."}
    finally:
        conn.close()

class SignupRequest(BaseModel):
    fullname: str
    email: str
    password: str
    role: str = "loan_officer"

@app.post("/api/auth/signup")
def signup_public(payload: SignupRequest):
    email_clean = payload.email.strip().lower()
    if not email_clean:
        raise HTTPException(status_code=400, detail="Email is required.")
        
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (email_clean,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="An account with this email already exists.")
            
        hashed = auth.hash_password(payload.password)
        cursor.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (email_clean, hashed, payload.role)
        )
        conn.commit()
        return {"success": True, "message": "Account created successfully."}
    finally:
        conn.close()

@app.post("/api/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    return {"message": "Logged out successfully."}

@app.post("/api/auth/forgot-password")
def forgot_password(payload: ForgotPasswordRequest):
    email_clean = payload.email.strip().lower()
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE username = ?", (email_clean,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="Email not found in our database.")
        
        # Real SMTP dispatch
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        
        if not sender_email or not sender_password:
            raise HTTPException(
                status_code=400,
                detail="SMTP credentials (SENDER_EMAIL/SENDER_PASSWORD) are not set in the server .env file. Please add them to send real emails."
            )
            
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        try:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email_clean
            msg['Subject'] = "RiskLens AI - Password Reset Request"
            
            body = f"""Hello,

You requested a password reset for your RiskLens AI account.

To reset your password, please navigate to:
http://localhost:8000/reset-password?email={email_clean}

If you did not make this request, you can safely ignore this email.

Best regards,
RiskLens AI Team"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to SMTP
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, email_clean, msg.as_string())
            server.quit()
            
            print(f"Real password reset email sent to: {email_clean}")
            return {"message": f"Password reset link has been successfully generated and sent to {email_clean}."}
        except Exception as e:
            print(f"SMTP error details: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"SMTP server failed to send email: {str(e)}"
            )
    finally:
        conn.close()

@app.get("/api/auth/me")
def get_me(user: dict = Depends(get_current_user)):
    return user

@app.get("/applicants/search")
@app.get("/api/applicants/search")
def search_applicants(query: Optional[str] = None, q: Optional[str] = None, user: dict = Depends(get_current_user)):
    search_term = query or q or ""
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        q_wildcard = "%" + search_term + "%"
        cursor.execute("""
            SELECT applicant_id, name, target 
            FROM applicants 
            WHERE CAST(applicant_id AS TEXT) LIKE ? OR name LIKE ? 
            LIMIT 6
        """, (search_term + "%", q_wildcard))
        rows = cursor.fetchall()
        return [{"id": r[0], "name": r[1], "target": r[2]} for r in rows]
    finally:
        conn.close()

@app.get("/applicants/{id}")
@app.get("/api/applicants/lookup/{id}")
def lookup_applicant(id: int, user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM applicants WHERE applicant_id = ?", (id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Applicant ID not found.")
        
        applicant_dict = dict(row)
        target = applicant_dict.pop("target", None)
        applicant_id = applicant_dict.pop("applicant_id", None)
        name = applicant_dict.pop("name", "Unknown Applicant")
        
        features = {}
        for col in X_test.columns:
            val = applicant_dict.get(col, None)
            features[col] = float(val) if val is not None and not pd.isna(val) else None
            
        return {
            "applicant_id": applicant_id,
            "name": name,
            "target": target,
            "features": features
        }
    finally:
        conn.close()

@app.get("/applicants/{id}/predict")
@app.get("/api/applicant/{id}")
def run_default_predict(id: int, user: dict = Depends(get_current_user)):
    if model is None or explainer is None or X_test is None:
        raise HTTPException(status_code=503, detail="Model or assets not loaded yet.")
        
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM applicants WHERE applicant_id = ?", (id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Applicant ID not found.")
            
        applicant_dict = dict(row)
        name = applicant_dict.pop("name", "Unknown Applicant")
        
        if id in X_test.index:
            row_series = X_test.loc[id].copy()
        else:
            row_series = X_test_mean.copy()
            
        row_df = pd.DataFrame([row_series])
        for col in row_df.columns:
            row_df[col] = row_df[col].astype(X_test[col].dtype)
            
        prob = float(model.predict_proba(row_df)[0, 1])
        shap_explanation = explainer(row_df)
        
        feature_names = list(row_df.columns)
        shap_values = shap_explanation.values[0]
        raw_values = row_df.values[0]
        
        explanation = []
        shap_values_dict = {}
        for name_feat, shap_val, raw_val in zip(feature_names, shap_values, raw_values):
            explanation.append({
                "feature": name_feat,
                "shap_value": float(shap_val),
                "feature_value": float(raw_val) if not np.isnan(raw_val) else None
            })
            shap_values_dict[name_feat] = float(shap_val)
            
        explanation = sorted(explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        narration = generate_narration(prob, explanation)
        prediction_id = log_prediction(id, user["id"], prob, shap_values_dict, narration=narration)
        
        return {
            "applicant_id": id,
            "name": name,
            "probability": prob,
            "base_value": float(shap_explanation.base_values[0]),
            "explanation": explanation,
            "prediction_id": prediction_id,
            "narration": narration
        }
    finally:
        conn.close()

@app.post("/applicants/{id}/predict")
@app.post("/api/applicant/{id}/recalculate")
def recalculate_applicant(id: int, payload: RecalculateRequest, user: dict = Depends(get_current_user)):
    if model is None or explainer is None or X_test is None:
        raise HTTPException(status_code=503, detail="Model or assets not loaded yet.")
        
    input_data = payload.features
    
    if id in X_test.index:
        row_series = X_test.loc[id].copy()
    else:
        row_series = X_test_mean.copy()
        
    for col, val in input_data.items():
        if col in row_series.index:
            row_series[col] = val
            
    row_df = pd.DataFrame([row_series])
    for col in row_df.columns:
        row_df[col] = row_df[col].astype(X_test[col].dtype)
        
    prob = float(model.predict_proba(row_df)[0, 1])
    shap_explanation = explainer(row_df)
    
    feature_names = list(row_df.columns)
    shap_values = shap_explanation.values[0]
    raw_values = row_df.values[0]
    
    explanation = []
    shap_values_dict = {}
    for name_feat, shap_val, raw_val in zip(feature_names, shap_values, raw_values):
        explanation.append({
            "feature": name_feat,
            "shap_value": float(shap_val),
            "feature_value": float(raw_val) if not np.isnan(raw_val) else None
        })
        shap_values_dict[name_feat] = float(shap_val)
        
    explanation = sorted(explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
    
    narration = generate_narration(prob, explanation)
    prediction_id = log_prediction(id, user["id"], prob, shap_values_dict, narration=narration)
    
    return {
        "applicant_id": id,
        "probability": prob,
        "base_value": float(shap_explanation.base_values[0]),
        "explanation": explanation,
        "prediction_id": prediction_id,
        "narration": narration
    }

@app.post("/decisions")
@app.post("/api/applicant/{id}/decision")
def record_decision(payload: DecisionRequest, user: dict = Depends(get_current_user), id: Optional[int] = None):
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT applicant_id FROM predictions WHERE id = ?", (payload.prediction_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction ID not found.")
        
        cursor.execute("""
            INSERT INTO decisions (prediction_id, officer_id, decision, notes)
            VALUES (?, ?, ?, ?)
        """, (payload.prediction_id, user["id"], payload.decision, payload.notes))
        conn.commit()
        return {"success": True, "decision_id": cursor.lastrowid}
    finally:
        conn.close()

@app.get("/applicants/{id}/history")
@app.get("/api/applicant/{id}/history")
def get_applicant_history(id: int, user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT p.id as prediction_id, p.timestamp as pred_time, p.predicted_probability, p.risk_label, p.narration,
                   d.id as decision_id, d.decision, d.notes, d.timestamp as decision_time,
                   u.username as officer_name
            FROM predictions p
            LEFT JOIN decisions d ON d.prediction_id = p.id
            LEFT JOIN users u ON d.officer_id = u.id
            WHERE p.applicant_id = ?
            ORDER BY p.timestamp DESC
        """, (id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

@app.get("/officers/{id}/activity")
def get_officer_activity(id: int, user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT d.id as decision_id, d.decision, d.notes, d.timestamp as decision_time,
                   p.id as prediction_id, p.applicant_id, p.predicted_probability, p.timestamp as pred_time,
                   a.name as applicant_name
            FROM decisions d
            JOIN predictions p ON d.prediction_id = p.id
            JOIN applicants a ON p.applicant_id = a.applicant_id
            WHERE d.officer_id = ?
            ORDER BY d.timestamp DESC
        """, (id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

@app.get("/api/decisions/stats")
def get_decisions_stats(user: dict = Depends(get_current_user)):
    conn = sqlite3.connect("credit_risk.db")
    cursor = conn.cursor()
    try:
        threshold = 0.15
        
        cursor.execute("""
            SELECT 
                p.predicted_probability,
                d.decision
            FROM decisions d
            JOIN predictions p ON d.prediction_id = p.id
        """)
        rows = cursor.fetchall()
        
        total = len(rows)
        if total == 0:
            return {
                "total_decisions": 0,
                "approved_count": 0,
                "declined_count": 0,
                "escalated_count": 0,
                "overrides_count": 0,
                "override_rate": 0.0
            }
            
        approved = 0
        declined = 0
        escalated = 0
        overrides = 0
        
        for prob, decision in rows:
            model_approved = prob < threshold
            officer_decision = decision.lower()
            
            if officer_decision == "approved":
                approved += 1
                if not model_approved:
                    overrides += 1
            elif officer_decision == "declined":
                declined += 1
                if model_approved:
                    overrides += 1
            elif officer_decision == "escalated":
                escalated += 1
                
        return {
            "total_decisions": total,
            "approved_count": approved,
            "declined_count": declined,
            "escalated_count": escalated,
            "overrides_count": overrides,
            "override_rate": float(overrides) / total if total > 0 else 0.0
        }
    finally:
        conn.close()

from typing import List

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    applicant_id: Optional[int] = None
    messages: List[Message]

# --- Chatbot Database & Model Tools ---

def get_applicant_with_prediction(applicant_id: int):
    """
    Fetches the profile and latest credit risk prediction for the applicant.
    """
    if model is None or X_test is None:
        return {"error": "Model or test assets not loaded."}
        
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM applicants WHERE applicant_id = ?", (applicant_id,))
        row = cursor.fetchone()
        if not row:
            return {"error": f"Applicant ID {applicant_id} not found."}
            
        applicant_dict = dict(row)
        name = applicant_dict.pop("name", "Unknown Applicant")
        target = applicant_dict.pop("target", None)
        
        if applicant_id in X_test.index:
            row_series = X_test.loc[applicant_id].copy()
        else:
            row_series = X_test_mean.copy()
            
        row_df = pd.DataFrame([row_series])
        for col in row_df.columns:
            row_df[col] = row_df[col].astype(X_test[col].dtype)
            
        prob = float(model.predict_proba(row_df)[0, 1])
        shap_explanation = explainer(row_df)
        
        feature_names = list(row_df.columns)
        shap_values = shap_explanation.values[0]
        raw_values = row_df.values[0]
        
        explanation = []
        for name_feat, shap_val, raw_val in zip(feature_names, shap_values, raw_values):
            display_name = FEATURE_DISPLAY_NAMES.get(name_feat, name_feat.replace("_", " ").title())
            explanation.append({
                "feature": name_feat,
                "display_name": display_name,
                "shap_value": float(shap_val),
                "feature_value": float(raw_val) if not np.isnan(raw_val) else None
            })
            
        explanation = sorted(explanation, key=lambda x: abs(x["shap_value"]), reverse=True)
        
        # Extract key financial and application metrics for LLM visibility
        key_metrics = {}
        for col in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_BIRTHDAY", "DAYS_EMPLOYED"]:
            if col in row_series.index:
                val = row_series[col]
                if val is not None and not pd.isna(val):
                    if col in ["DAYS_BIRTH", "DAYS_BIRTHDAY"]:
                        key_metrics[col] = {
                            "raw_value": float(val),
                            "value_in_years": round(abs(float(val)) / 365.25, 1)
                        }
                    elif col == "DAYS_EMPLOYED":
                        key_metrics[col] = {
                            "raw_value": float(val),
                            "value_in_years": round(abs(float(val)) / 365.25, 1) if float(val) < 0 else 0
                        }
                    else:
                        key_metrics[col] = float(val)

        return {
            "applicant_id": applicant_id,
            "name": name,
            "actual_repayment_status": "Paid" if target == 0 else ("Defaulted" if target == 1 else "Unknown"),
            "predicted_default_probability": prob,
            "risk_classification": "High Risk" if prob >= 0.25 else ("Moderate Risk" if prob >= 0.10 else "Low Risk"),
            "key_application_metrics": key_metrics,
            "top_shap_risk_drivers": explanation[:10]
        }
    finally:
        conn.close()

def run_model_with_changes(applicant_id: int, changes: dict):
    """
    Simulates a what-if scenario by changing features on top of an applicant's profile and re-predicting.
    """
    if model is None or X_test is None:
        return {"error": "Model or test assets not loaded."}
        
    if applicant_id in X_test.index:
        row_series = X_test.loc[applicant_id].copy()
    else:
        row_series = X_test_mean.copy()
        
    # Override changes
    for col, val in changes.items():
        if col in row_series.index:
            row_series[col] = val
            
    row_df = pd.DataFrame([row_series])
    for col in row_df.columns:
        row_df[col] = row_df[col].astype(X_test[col].dtype)
        
    prob = float(model.predict_proba(row_df)[0, 1])
    
    simulated_changes = {}
    for col in changes.keys():
        if col in X_test.columns:
            orig_val = X_test.loc[applicant_id, col] if applicant_id in X_test.index else X_test_mean[col]
            simulated_changes[col] = {
                "feature": col,
                "display_name": FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title()),
                "original_value": float(orig_val) if not pd.isna(orig_val) else None,
                "new_value": float(changes[col])
            }
            
    return {
        "applicant_id": applicant_id,
        "new_predicted_default_probability": prob,
        "new_risk_classification": "High Risk" if prob >= 0.25 else ("Moderate Risk" if prob >= 0.10 else "Low Risk"),
        "simulated_changes": list(simulated_changes.values())
    }

def fetch_decision_history(applicant_id: int):
    """
    Retrieves all past predictions and officer decisions for the applicant.
    """
    conn = sqlite3.connect("credit_risk.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT p.id as prediction_id, p.timestamp as prediction_time, p.predicted_probability, p.risk_label,
                   d.decision, d.notes, d.timestamp as decision_time,
                   u.username as officer_name
            FROM predictions p
            LEFT JOIN decisions d ON d.prediction_id = p.id
            LEFT JOIN users u ON d.officer_id = u.id
            WHERE p.applicant_id = ?
            ORDER BY p.timestamp DESC
        """, (applicant_id,))
        rows = cursor.fetchall()
        
        history_list = []
        for r in rows:
            history_list.append({
                "prediction_id": r["prediction_id"],
                "prediction_time": r["prediction_time"],
                "predicted_probability": r["predicted_probability"],
                "risk_label": r["risk_label"],
                "decision": r["decision"] or "Awaiting decision",
                "notes": r["notes"] or "",
                "decision_time": r["decision_time"] or "",
                "officer_name": r["officer_name"] or ""
            })
        return history_list
    finally:
        conn.close()

def execute_tool(tool_name: str, arguments: dict, applicant_id: int):
    target_id = arguments.get("applicant_id", applicant_id)
    try:
        target_id = int(target_id)
    except (ValueError, TypeError):
        pass
    if tool_name == "get_applicant_data":
        return get_applicant_with_prediction(target_id)
    elif tool_name == "run_what_if":
        return run_model_with_changes(target_id, arguments.get("changes", {}))
    elif tool_name == "get_decision_history":
        return fetch_decision_history(target_id)
    return {"error": f"Unknown tool: {tool_name}"}

# --- OpenAI Function-Calling Tool Specifications ---

tools_spec = [
    {
        "type": "function",
        "function": {
            "name": "get_applicant_data",
            "description": "Get the full profile and latest risk prediction for an applicant. This includes key_application_metrics such as requested loan/credit amount (AMT_CREDIT), monthly/annual annuity (AMT_ANNUITY), applicant total annual income (AMT_INCOME_TOTAL), and purchase price (AMT_GOODS_PRICE).",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer",
                        "description": "The applicant ID"
                    }
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_what_if",
            "description": "Re-run the risk model with modified feature values to simulate a what-if scenario",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer"
                    },
                    "changes": {
                        "type": "object",
                        "description": "Key-value pairs of features to change, e.g. {\"EXT_SOURCE_2\": 0.8, \"AMT_GOODS_PRICE\": 500000}"
                    }
                },
                "required": ["applicant_id", "changes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_decision_history",
            "description": "Get all past predictions and officer decisions for an applicant",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {
                        "type": "integer"
                    }
                },
                "required": ["applicant_id"]
            }
        }
    }
]

@app.post("/chat")
@app.post("/api/chat")
async def chat(request: ChatRequest, user: dict = Depends(get_current_user)):
    system_prompt = f"""You are RiskLens AI, an assistant for bank loan officers reviewing credit applications.

You have access to three tools:
- get_applicant_data: fetch a full applicant profile and their current risk score
- run_what_if: re-run the model with hypothetical changes to see how risk would shift
- get_decision_history: retrieve past predictions and officer decisions for this applicant

Model Context & Database Info:
- The model is a CatBoost binary classifier trained on a historical database of 307,511 loan applications.
- Performance: Evaluated on a separate holdout test set (represented by the applicants in the database) achieving a ROC-AUC score of 0.78.
- Calibration: The model's raw probabilities are well-calibrated (mean gap < 4%), meaning a predicted 20% default risk reflects an empirical ~20% default rate in historical data.
- Database Records: The `actual_repayment_status` (Paid vs Defaulted) is the true historical target label of the applicant in the test set. The model's default probability predicts risk for a new/hypothetical loan application of similar characteristics.

{FEATURE_MAPPING_TEXT}

Rules you must follow:
1. Never state a final approve or decline recommendation. Your job is to explain risk factors, not make the decision.
2. Never invent numbers. Always call the appropriate tool to get real model outputs.
3. Keep answers concise — 2-4 sentences unless the officer asks for more detail.
4. When citing any feature or risk factor, you MUST use the Plain English Name from the Feature Name Reference above. Never use raw column names like EXT_SOURCE_2, AMT_CREDIT, or DAYS_EMPLOYED in your responses.
5. The current applicant being reviewed is applicant ID {request.applicant_id}.
"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Rebuild incoming messages list
    for m in request.messages:
        msg_dict = {"role": m.role}
        if m.content is not None:
            msg_dict["content"] = m.content
        if m.tool_calls is not None:
            msg_dict["tool_calls"] = m.tool_calls
        messages.append(msg_dict)

    # 1. First Call to OpenAI / Azure OpenAI Endpoint
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    model_name = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    
    if not openai_key:
        # Fallback local response if no keys exist
        return {"reply": "API Key is missing in server environment. Unable to run chatbot."}

    import requests
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    req_payload = {
        "model": model_name,
        "messages": messages,
        "tools": tools_spec,
        "tool_choice": "auto"
    }
    
    if "gpt-5" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower():
        req_payload["max_completion_tokens"] = 4000
    else:
        req_payload["max_tokens"] = 4000
        req_payload["temperature"] = 0.2

    try:
        print(f"Chat API Request: applicant_id={request.applicant_id}, messages_len={len(messages)}")
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=req_payload, timeout=25)
        print(f"First turn response status: {response.status_code}")
        if response.status_code != 200:
            print(f"First turn failed: {response.text}")
            return {"reply": f"API Error (status {response.status_code}): {response.text}"}
            
        res_json = response.json()
        choice_message = res_json["choices"][0]["message"]
        print(f"First turn message: {json.dumps(choice_message)}")
        
        # If the LLM requests a tool call
        if "tool_calls" in choice_message and choice_message["tool_calls"]:
            tool_call = choice_message["tool_calls"][0]
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            print(f"Executing tool {tool_name} with args {arguments}")
            
            # Execute local tool function
            tool_result = execute_tool(tool_name, arguments, request.applicant_id)
            print(f"Tool result size: {len(str(tool_result))}")
            
            # Append LLM's tool call suggestion
            messages.append({
                "role": "assistant",
                "content": choice_message.get("content"),
                "tool_calls": choice_message["tool_calls"]
            })
            
            # Append the tool execution result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": json.dumps(tool_result)
            })
            
            # 2. Second Call to LLM with the tool result included
            second_payload = {
                "model": model_name,
                "messages": messages,
                "tools": tools_spec
            }
            if "gpt-5" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower():
                second_payload["max_completion_tokens"] = 4000
            else:
                second_payload["max_tokens"] = 4000
                second_payload["temperature"] = 0.2
                
            print(f"Sending second turn to LLM. Messages count: {len(messages)}")
            second_resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=second_payload, timeout=25)
            print(f"Second turn response status: {second_resp.status_code}")
            if second_resp.status_code != 200:
                print(f"Second turn failed: {second_resp.text}")
                return {"reply": f"API Error on second turn (status {second_resp.status_code}): {second_resp.text}"}
                
            second_json = second_resp.json()
            print(f"Second turn response json: {json.dumps(second_json, indent=2)}")
            reply = second_json["choices"][0]["message"]["content"].strip() if second_json["choices"][0]["message"].get("content") else ""
            print(f"Second turn reply: {reply.encode('ascii', errors='replace').decode('ascii')}")
            return {"reply": reply}
            
        else:
            reply = choice_message["content"].strip() if choice_message.get("content") else ""
            print(f"Direct response (no tools) reply: {reply.encode('ascii', errors='replace').decode('ascii')}")
            return {"reply": reply}
            
    except Exception as e:
        print("Chatbot execution error:", e)
        import traceback
        traceback.print_exc()
        return {"reply": f"Internal chatbot processing error: {str(e)}"}

@app.get("/api/model/calibration")
@app.get("/model/calibration")
def get_calibration():
    try:
        with open(STATIC_DIR / "calibration_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Calibration data not generated yet.")

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
