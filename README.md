# Explainable Credit Card Fraud Detection and Risk Scoring

An end-to-end, enterprise-ready machine learning system that identifies fraudulent credit card transactions, produces calibrated risk scores, and delivers local and global model interpretability using SHAP and interactive dashboards.

---

## Business Objective
Credit card fraud costs financial institutions billions annually. Detecting fraud presents distinct machine learning challenges:
1. **Severe Class Imbalance**: Legitimate transactions outnumber fraudulent events by ~580:1 (~0.172% fraud rate).
2. **False Positive Fatigue**: Overly aggressive flags hurt user trust and surge manual review costs.
3. **Regulatory & Audit Demands**: Black-box predictions are unacceptable in regulated financial services; investigators need explainable drivers for every flagged transaction.

This project delivers:
- Fast in-memory analytics using **DuckDB**.
- Optimized gradient boosting models (**XGBoost**) calibrated with cost-sensitive decision thresholds.
- Feature attribution and interpretability via **SHAP**.
- An analyst-facing **Streamlit** dashboard for transaction monitoring and real-time risk scoring.

---

## Architecture & Structure

```
fraud-risk-intelligence/
├── AGENTS.md                 # Agent guidelines and 7-day execution plan
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore specifications
├── data/
│   ├── raw/                  # Original data files (untracked)
│   ├── processed/            # Feature-engineered Parquet files
│   └── database/             # DuckDB analytical databases
├── notebooks/                # Exploratory analysis and prototyping
├── configs/                  # Model configurations and hyperparameter settings
├── artifacts/                # Serialized models, scalers, and evaluation reports
├── src/
│   ├── __init__.py
│   ├── ingestion/            # DuckDB data loading and conversion pipelines
│   ├── features/             # Scalers, transformations, and feature pipelines
│   ├── models/               # Training routines, cross-validation, threshold tuning
│   ├── explainability/       # SHAP explainers, risk scoring engine
│   └── app/                  # Streamlit monitoring and scoring UI
└── tests/
    ├── __init__.py
    └── test_environment.py   # Environment and dependency verification tests
```

---

## Tech Stack
- **Runtime**: Python 3.12
- **Data Engine**: DuckDB, Pandas, NumPy
- **Modeling**: scikit-learn, XGBoost
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Visualization & UI**: Plotly, Streamlit
- **Testing**: pytest

---

## Getting Started

### 1. Prerequisites
Ensure Python 3.12 is installed on your system.

### 2. Environment Setup
```bash
# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Verification Tests
```bash
pytest tests/ -v
```
