# RiskLens AI — Explainable Credit Risk Dashboard & ML Pipeline

[![Live Site](https://img.shields.io/badge/Live%20Demo-Render-brightgreen?style=for-the-badge)](https://risklens-ai-2e1e.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Gemini%20AI-blue?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![ML Pipeline](https://img.shields.io/badge/CatBoost--1.2.10-red?style=for-the-badge)](https://catboost.ai)

An end-to-end, production-grade credit underwriting and risk assessment application. RiskLens AI combines a high-performance **CatBoost** binary classifier (trained on the 307K-record Home Credit dataset) with an interactive **FastAPI** web dashboard. It features **SHAP explainability**, dynamic **what-if simulation**, and an **LLM-powered underwriting assistant** to help loan officers interpret risk factors instantly and audit decision-making.

---

## 🚀 Live Demo & Key Features

👉 **Explore the Live Dashboard:** [https://risklens-ai-2e1e.onrender.com](https://risklens-ai-2e1e.onrender.com)

*   **Interactive Underwriter Dashboard**: View pending loan applicants, sort by default risk, and review historical decisions.
*   **Local SHAP Explainability**: Visualizes the exact positive/negative contributions of applicant features (e.g., Credit Bureau scores, debt-to-income, repayment delays) to their credit score.
*   **Dynamic What-If Analysis**: Change applicant features (like decreasing loan annuity or increasing employment duration) and instantly recalculate default probability in real-time.
*   **LLM Underwriting Assistant (Gemini 2.5 Flash)**: Generates a human-friendly, 2-sentence risk summary identifying the top 3 credit risk factors in plain English, preventing loan officers from getting lost in raw features.
*   **JWT Secure Session Management**: Includes authentication, role-based access control (Admin vs. Loan Officer), and a decision audit trail saved to an SQLite database.

---

## 🛠️ Tech Stack & Architecture

```
        +-------------------------------------------------------+
        |                 Frontend (HTML5/Vanilla CSS/JS)       |
        +-------------------------------------------------------+
                                    | (JWT Authentication / REST API)
                                    v
        +-------------------------------------------------------+
        |                 FastAPI Application Server            |
        +-------------------------------------------------------+
           |                         |                        |
           v (Predict / SHAP)        v (Narration Prompt)     v (Audit Trail)
  +------------------+      +-------------------+     +-----------------+
  |  CatBoost Model  |      |   Gemini 2.5      |     |  SQLite DB      |
  |  & SHAP Engine   |      |   Flash API       |     |  (credit_risk)  |
  +------------------+      +-------------------+     +-----------------+
```

*   **Backend**: FastAPI, Uvicorn, Python 3.12 (asynchronous, high-performance API server).
*   **Frontend**: Vanilla HTML5, CSS3 (Modern Glassmorphism & Dark Mode UI), and JavaScript.
*   **Database**: SQLite (local schema tracking Users, Sessions, Predictions, and Decisions).
*   **AI/LLM**: Google GenAI SDK (Gemini 2.5 Flash for natural language underwriting support).
*   **Deployment**: Docker-packaged and deployed on **Render Free Web Service**, kept warm using a **GitHub Actions keep-warm ping workflow** running every 10 minutes.

---

## 📊 Machine Learning Pipeline

The backend is backed by an end-to-end reproducible machine learning pipeline (available in the `notebooks/` directory) trained on Kaggle's **Home Credit Default Risk** dataset:

1.  **Business Understanding & EDA**: Analyzed extreme class imbalance (~91.8% repaid, ~8.2% default) and identified feature distributions.
2.  **Feature Engineering**: Created custom interaction metrics including `EXT_SOURCE_MEAN`, `CREDIT_INCOME_RATIO`, and `CREDIT_TERM` along with transactional aggregations (POS Cash, credit card utilization trends, delinquency delays).
3.  **Automated Feature Selection**: Combined 4 feature importance methods (Lasso L1, XGBoost Importance, Target Correlation, and Recursive Feature Elimination) to filter down to the most robust predictors.
4.  **Ensemble Tuning**: Tuned LightGBM, XGBoost, and CatBoost models. CatBoost was dynamically selected as the best classifier.
5.  **Threshold Optimization**: Shifted the decision threshold from `0.50` to `0.15` to optimize the F1-score, reducing False Negatives (unapproved defaulters) and maximizing credit profit metrics.

### Model Metrics (Holdout Test Set)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.781** | Strong discriminative capability |
| **Recall** | **0.450** | Catches 45% of actual defaults (up from ~7% at a 0.50 threshold) |
| **F1 Score** | **0.334** | Optimal balance for the highly imbalanced target |

---

## ⚙️ Installation & Local Setup

### Prerequisites
Make sure Python (>= 3.12) is installed.

### Setup Environment
1.  Clone the repository:
    ```bash
    git clone https://github.com/PalakGoyal6/RiskLens-AI.git
    cd RiskLens-AI
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Initialize Database
Run the setup script to generate the SQLite database and seed initial test users and applicant records:
```bash
python init_db.py
```

### Start Development Server
```bash
uvicorn server:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🔒 Environment Variables
Create a `.env` file in the root directory to store your API keys and secrets:
```env
GEMINI_API_KEY=your-gemini-api-key-here
JWT_SECRET=super-secret-key-for-credit-risk-token-signing
```

---

## 📈 Pipeline Development Notebooks
If you want to view or compile the Jupyter Notebooks for the model pipeline:
```bash
# Compile Python scripts into Jupyter Notebooks
python build_nb01.py
python build_nb02.py
python build_nb03.py
python build_nb04.py
```
Notebooks will be generated in the `notebooks/` directory for step-by-step exploration.
