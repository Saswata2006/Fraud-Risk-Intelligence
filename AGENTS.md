# AGENTS.md - Multi-Agent Operating Guidelines & Rules

## Project Identity
- **Project**: Explainable Credit Card Fraud Detection and Risk Scoring
- **Objective**: Build an internship-ready, enterprise-grade Data Science & ML solution in 7 days.
- **Dataset**: Kaggle ULB Credit Card Fraud Detection (284,807 transactions, highly imbalanced at ~0.172% fraud).
- **Core Tech Stack**: Python 3.12, Pandas, NumPy, DuckDB, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, Pytest.

---

## 7-Day Execution Blueprint

| Day | Focus Area | Deliverables |
| :--- | :--- | :--- |
| **Day 1** | **Foundation & Architecture** | Repository structure, clean virtualenv (Python 3.12), test harness, agent rules, and documentation. |
| **Day 2** | **Data Ingestion & In-Memory Analytics** | DuckDB ingestion pipeline, Parquet conversion, SQL queries for EDA, class distribution profiling. |
| **Day 3** | **Feature Engineering & Imbalance Handling** | Robust scaling (Amount, Time), cyclical feature transforms, PR-AUC evaluation setup, sampling/weighting strategies. |
| **Day 4** | **Model Development & Threshold Tuning** | Logistic Regression benchmark vs XGBoost, PR-AUC optimization, cost-sensitive threshold calibration. |
| **Day 5** | **Explainability & Model Governance** | SHAP TreeExplainer integration, global summary plots, local transaction waterfall/force plots, risk tiered scoring. |
| **Day 6** | **Interactive Streamlit Dashboard** | Live transaction scoring UI, analyst review queue, interactive SHAP visualizer with Plotly. |
| **Day 7** | **Hardening, Testing & Portfolio Delivery** | End-to-end integration tests, benchmarking report, production-grade documentation, interview talk-track. |

---

## Operating Rules for Agents

### 1. Data Integrity & Scientific Rigor
- **Zero Hallucination of Results**: Never make up metrics, benchmark scores, or model evaluation outputs. All metrics presented must be computed from reproducible code runs.
- **Strict Leakage Prevention**: Fit scalers, transformers, and imputers ONLY on the training split. Never fit on test/validation data.
- **Evaluation Metric Alignment**: In high class-imbalance fraud scenarios, do NOT rely on standard accuracy or simple ROC-AUC. Primary metrics must be **PR-AUC (Average Precision)**, **Recall@Precision >= 0.80**, and **Expected Financial Loss reduction**.

### 2. Code Modularity & Production Standards
- **No Monoliths**: All business and ML logic must live under `src/` organized by responsibility:
  - `src/ingestion`: DuckDB and data loading routines.
  - `src/features`: Transformations, pipelines, scaling.
  - `src/models`: Model training, hyperparameter tuning, threshold calibration.
  - `src/explainability`: SHAP explainer utilities and risk score categorization.
  - `src/app`: Streamlit front-end and dashboard components.
- **Type Annotations & Docstrings**: All core functions and classes must include clear type hints and Google/NumPy-style docstrings.
- **Reproducibility**: Explicitly seed all pseudo-random generators (`random_state=42`).

### 3. Safety & Command Execution
- Stay strictly within `/Users/Saswata007/Desktop/fraud-risk-intelligence`.
- Do not download datasets, train long models, or push to remotes without explicit user instruction.
- Never issue destructive commands (`rm -rf`, force resets) without confirmation.
- Keep tests updated: run `pytest` to verify changes after significant implementations.
