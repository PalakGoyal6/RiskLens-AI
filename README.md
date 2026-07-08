# RiskLens AI - Explainable Credit Risk Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen?style=for-the-badge)](https://risklens-ai-2e1e.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/GPT--5%20mini-black?style=for-the-badge&logo=openai)](https://openai.com)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-red?style=for-the-badge)](https://catboost.ai)

An end-to-end credit risk prediction platform built for loan officers. RiskLens AI scores loan applicants on default probability, explains every decision with SHAP, generates plain-English AI narrations, and maintains a full officer audit trail- deployed and live at [risklens-ai-2e1e.onrender.com](https://risklens-ai-2e1e.onrender.com).

> Credit default prediction is one of the most consequential applications of ML in finance. This project goes beyond model accuracy to build the full decision-support system a real bank would need- explainability, auditability, role-based access, and a user interface loan officers can actually use.

<p align="center">
<img src="https://github.com/user-attachments/assets/9df6dbe0-f931-456c-9a61-fd740054fd2d" width="47%">
<img src="https://github.com/user-attachments/assets/26420bbc-1fed-4d89-b2e3-02004600fd65" width="47%">
</p>

---

## Live Demo

**[risklens-ai-2e1e.onrender.com](https://risklens-ai-2e1e.onrender.com)** - click **"Continue as Demo User"** on the login page to explore without creating an account.

The demo environment is pre-seeded with applicant profiles, past predictions, and officer decisions so every feature of the product is immediately visible.

---

## Features

- **Real-time risk scoring** - CatBoost model scores any applicant instantly via REST API, achieving ROC-AUC of 0.781 on a holdout set of 61K+ loans
- **SHAP explainability** - global feature importance bar chart, beeswarm plot, and per-applicant waterfall chart showing exactly which factors drove each score
- **Live what-if simulation** - drag sliders to modify applicant features (credit scores, income, debt ratios) and watch the risk prediction and SHAP values update in real time
- **AI underwriting assistant** - GPT-5 mini chatbot with function-calling for applicant lookup, what-if simulation, and decision history queries - scoped strictly to the open applicant
- **AI decision narration** - auto-generated 2-sentence plain-English summary of every prediction, stored alongside the score in the audit log
- **Full audit trail** - every prediction, officer decision, override note, and AI narration logged with timestamp and officer ID
- **Auth + role-based access** - JWT-based authentication with role separation: Loan Officer, Risk Analyst, Admin
- **Probability calibration validated** - reliability diagram confirms raw CatBoost probabilities are well-calibrated (mean gap < 4%); Platt scaling evaluated and deliberately not applied

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | CatBoost (tuned via RandomizedSearchCV), SHAP |
| Backend | FastAPI, Python 3.12, Uvicorn |
| Frontend | Vanilla HTML5, CSS3, JavaScript |
| Database | SQLite |
| Auth | Custom JWT (python-jose) |
| AI Layer | OpenAI GPT-5 mini (function-calling) |
| Deployment | Docker on Render |

---

## Architecture

```
+-------------------------------------------------------+
|         Frontend (HTML5 / CSS3 / Vanilla JS)          |
|  Landing · Login · Dashboard · SHAP · Chatbot · Audit |
+-------------------------------------------------------+
                        |
            JWT Authentication / REST API
                        |
                        v
+-------------------------------------------------------+
|              FastAPI Application Server               |
+-------------------------------------------------------+
        |                    |                   |
        v                    v                   v
+---------------+   +----------------+   +----------------+
| CatBoost Model|   | OpenAI GPT-5  |   | SQLite DB      |
| + SHAP Engine |   | mini (Chatbot  |   | Users ·        |
|               |   | + Narration)   |   | Predictions ·  |
+---------------+   +----------------+   | Decisions      |
                                         +----------------+
```

---

## ML Pipeline

End-to-end reproducible pipeline across 4 notebooks (see `notebooks/` directory), trained on Kaggle's **Home Credit Default Risk** dataset (307,511 loan applications across 8 relational tables).

### Pipeline Stages

**1. Business Understanding & EDA**
Analyzed extreme class imbalance (~91.8% repaid, ~8.2% default), missing value profiles, and numeric distributions. Identified anomalies including `DAYS_EMPLOYED = 365243` encoding missing employment as a sentinel value.

**2. Feature Engineering**
Computed customer-level aggregations from transactional tables (POS Cash, credit card utilization, installment delays, bureau delinquency trends). Created interaction features including `EXT_SOURCE_MEAN`, `CREDIT_INCOME_RATIO`, and `CREDIT_TERM`. Expanded from 120 raw variables to 215+ engineered features.

**3. Automated Feature Selection**
Filtered 215+ candidate features using 4 independent methods - Lasso L1 regularization, XGBoost feature importance, target correlation analysis, and Recursive Feature Elimination. Retained features selected by at least 2 of 4 methods to maximize generalization.

**4. Model Training & Tuning**
Benchmarked Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost. Tuned top models via `RandomizedSearchCV`. CatBoost dynamically selected as best by cross-validated ROC-AUC.

**5. Threshold Optimization & Calibration**
Shifted classification threshold from default 0.50 to 0.15 to account for class imbalance, improving recall from ~7% to 45%. Validated probability calibration via reliability diagram - confirmed mean gap < 4%, so raw probabilities were retained as the most accurate and principled choice.

### Model Performance (Holdout Test Set - 20% split, never seen during training)

| Metric | Score | Note |
|---|---|---|
| **ROC-AUC** | **0.781** | Strong discriminative capability across all thresholds |
| **Recall** | **0.450** | Catches 45% of actual defaulters (up from ~7% at threshold 0.50) |
| **Precision** | 0.266 | Low false-alarm rate relative to class imbalance |
| **F1 Score** | 0.334 | Optimal balance for 8.2% positive class rate |

### Baseline vs. Tuned Model Comparison (5-Fold Cross-Validation)

| Model | Baseline ROC-AUC | Tuned ROC-AUC |
|---|---|---|
| **CatBoost** | 0.747 | **0.769** |
| XGBoost | 0.745 | 0.765 |
| LightGBM | 0.743 | 0.761 |
| Random Forest | 0.713 | - |
| Logistic Regression | 0.706 | - |

---

## Key Findings

**Credit bureau scores dominate default prediction.**
EXT_SOURCE_2, EXT_SOURCE_3, and EXT_SOURCE_1 were the top 3 features by mean absolute SHAP value (0.298, 0.275, and 0.151 respectively), dwarfing all engineered features. Bureau integration is the single highest-leverage data investment for any lender.

**Threshold optimization is the real business lever.**
At the default 0.50 threshold, the model caught only ~7% of defaulters. Shifting to 0.15 improved recall to 45% - identifying 6x more actual defaulters, directly reducing expected write-off losses at the cost of more conservative approvals.

**Platt scaling was evaluated and deliberately rejected.**
A reliability diagram confirmed raw CatBoost probabilities are already well-calibrated (mean gap < 4%). Applying Platt scaling increased the Brier score from 0.0661 to 0.0674 - making predictions less accurate. Raw probabilities were retained as the principled choice.

**Ensemble stacking added no value.**
A weighted ensemble of CatBoost, LightGBM, and XGBoost was evaluated - the optimizer assigned near-zero weight to LightGBM and XGBoost due to high inter-model correlation. CatBoost alone was retained.

---

## Screenshots

## Screenshots

<table>
<tr>
<td align="center">
<b>Dashboard</b><br><br>
<img src="https://github.com/user-attachments/assets/3a3ede1a-7d3c-4fa4-bc83-b391fb6003f5" width="450">
</td>

<td align="center">
<b>SHAP Explainability</b><br><br>
<img src="https://github.com/user-attachments/assets/d7225ad3-9a18-45e3-b16a-4a64e90a1678" width="450">
</td>
</tr>

<tr>
<td align="center">
<b>AI Chatbot</b><br><br>
<img src="https://github.com/user-attachments/assets/84e985bb-0b6f-49a4-9448-8a62c5fe7a4b" width="450">
</td>

<td align="center">
<b>Calibration Curve</b><br><br>
<img src="https://github.com/user-attachments/assets/e191fa4c-6811-4733-b90d-3e675079eb13" width="450">
</td>
</tr>
</table>

---

## Local Setup

### Prerequisites
Python >= 3.12

### Setup

```bash
git clone https://github.com/PalakGoyal6/RiskLens-AI.git
cd RiskLens-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Initialize Database

```bash
python init_db.py
```

Generates the SQLite database and seeds initial test users and applicant records.

### Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
JWT_SECRET=your_jwt_secret_key
```

### Start Development Server

```bash
uvicorn server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Run ML Pipeline Notebooks

```bash
python build_nb01.py
python build_nb02.py
python build_nb03.py
python build_nb04.py
```

Notebooks are generated in `notebooks/` for step-by-step pipeline exploration.

---

## Deployment

Dockerized and deployed on **Render Free Web Service**.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

A GitHub Actions workflow pings the service every 10 minutes to prevent cold starts on Render's free tier.

---

## Repository Structure

```
risklens-ai/
├── server.py                  # FastAPI app, endpoints, chatbot, narration
├── init_db.py                 # Database schema + demo data seeding
├── Dockerfile
├── requirements.txt
├── .env.example
├── static/
│   ├── index.html             # Main dashboard
│   ├── index.js               # Frontend logic, SHAP charts, chatbot UI
│   └── styles.css
├── models/
│   └── catboost_model.pkl
├── data/
│   └── calibration_data.json
├── notebooks/
│   ├── 01_BusinessUnderstanding_EDA.ipynb
│   ├── 02_DataCleaning_FeatureEngineering.ipynb
│   ├── 03_ModelTraining.ipynb
│   └── 04_Explainability_Calibration.ipynb
└── images/
    ├── dashboard.png
    ├── shap_waterfall.png
    ├── chatbot.png
    └── calibration.png
```

---

## Author

Built by **Palak Goyal** as a portfolio project demonstrating end-to-end ML engineering, full-stack deployment, and responsible AI practices in financial services.

- [LinkedIn](https://www.linkedin.com/in/palakgoyal2006/) 
- [GitHub](https://github.com/PalakGoyal6)
