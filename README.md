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
- Leakage-safe chronological splitting and preprocessing pipelines via **scikit-learn**.
- Baseline **Logistic Regression** benchmark with `class_weight='balanced'`.
- Optimized gradient boosting models (**XGBoost**) calibrated with cost-sensitive decision thresholds.
- Feature attribution and interpretability via **SHAP**.
- An analyst-facing **Streamlit** dashboard for transaction monitoring and real-time risk scoring.

---

## 7-Day Blueprint & Progress

| Day | Focus Area | Status | Deliverables |
| :--- | :--- | :--- | :--- |
| **Day 1** | **EDA, Ingestion & SQL Analytics** | **DONE** | Schema validation, DuckDB SQL analytics, 13 figures, data quality report, Jupyter EDA notebook. |
| **Day 2** | **Splitting, Feature Pipeline & Baseline** | **DONE** | Chronological 60/20/20 split, leakage audit, `log1p` + `RobustScaler` pipeline, Logistic Regression baseline. |
| **Day 3** | **Feature Engineering & Imbalance Handling** | PENDING | Cyclical time transforms, transaction velocity, PR-AUC evaluation setup, cost matrix. |
| **Day 4** | **Model Development & Threshold Tuning** | PENDING | XGBoost training vs Logistic Regression, PR-AUC optimization, cost-sensitive threshold calibration. |
| **Day 5** | **Explainability & Model Governance** | PENDING | SHAP TreeExplainer integration, waterfall/force plots, risk tiered scoring. |
| **Day 6** | **Interactive Streamlit Dashboard** | PENDING | Live transaction scoring UI, analyst review queue, interactive SHAP visualizer. |
| **Day 7** | **Hardening, Testing & Portfolio Delivery** | PENDING | End-to-end integration tests, benchmarking report, production documentation. |

---

## Architecture & Project Structure

```
fraud-risk-intelligence/
├── AGENTS.md                 # Agent guidelines and operating rules
├── README.md                 # Project documentation and quickstart
├── requirements.txt          # Python dependencies
├── .gitignore                # Production data science ignore patterns
├── data/
│   ├── raw/                  # Original data files (untracked, creditcard.csv)
│   ├── processed/            # Feature-engineered Parquet partitions
│   └── database/             # DuckDB analytical databases
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Day 1 EDA and SQL profiling
│   └── 02_baseline_modeling.ipynb           # Day 2 Splitting and Baseline modeling
├── reports/
│   ├── data_quality_report.md               # Data validation and audit report
│   └── figures/                             # High-resolution PNGs and interactive HTML plots
├── artifacts/
│   ├── baseline_logistic_regression.joblib  # Serialized baseline model pipeline
│   └── baseline_metrics.json                # Actual baseline validation metrics
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── data_loader.py    # Schema validation and dataset loader
│   │   └── sql_analytics.py  # DuckDB vectorized SQL queries
│   ├── features/
│   │   ├── data_split.py     # Chronological splitting and duplicate audit
│   │   ├── preprocessor.py   # Leakage-safe ColumnTransformer pipelines
│   │   └── eda_analysis.py   # Statistical summaries and visualization engine
│   ├── models/
│   │   ├── baseline.py       # Baseline Logistic Regression pipeline
│   │   └── evaluation.py     # Metrics calculation and diagnostic curves
│   ├── explainability/       # SHAP explainers, risk scoring engine (Day 5)
│   ├── app/                  # Streamlit monitoring and scoring UI (Day 6)
│   ├── run_day1_eda.py       # Day 1 execution script
│   └── run_day2_pipeline.py  # Day 2 execution script
└── tests/
    ├── test_environment.py       # Environment and dependencies tests
    ├── test_data_validation.py   # Ingestion and schema validation tests
    ├── test_sql_analytics.py     # DuckDB SQL analytical tests
    ├── test_eda_visualizations.py# Figure existence and non-empty tests
    ├── test_data_split.py        # Split bounds and cross-partition duplicate tests
    └── test_baseline_model.py    # Preprocessor, shape, probability, and metric tests
```

---

## Methodology & Engineering Decisions

### 1. Data Splitting Strategy (Chronological)
- Financial transaction streams exhibit natural temporal dependencies. A standard random split would scatter duplicate and burst transactions across partitions, leaking future customer and merchant patterns into training.
- Transactions are sorted chronologically by `Time` and split into:
  - **Train Partition (60%)**: 170,884 transactions | 360 frauds (0.2107% fraud rate)
  - **Validation Partition (20%)**: 56,961 transactions | 57 frauds (0.1001% fraud rate)
  - **Test Partition (20%)**: 56,962 transactions | 75 frauds (0.1317% fraud rate)
- **Test Set Protection**: The test set is completely held out and untouched. All model evaluation and threshold tuning occur exclusively on the validation set.

### 2. Cross-Partition Duplicate Leakage Audit
- Full-row duplicates (matching all 31 features including timestamp) across partition boundaries: **0**.
- Identical feature vectors $(V1-V28, Amount)$ occurring at different timestamps:
  - 274 feature profiles in Validation appeared in Training (561 rows total).
  - An audit confirmed **0 of these shared profile rows are fraudulent** (all are legitimate recurring subscription or common transactions).
  - Evaluating on the deduplicated validation subset yields identical performance (**PR-AUC = 0.7552**, **Recall = 0.8772**), proving the model does not depend on memorizing past profiles.

### 3. Leakage-Safe Feature Engineering Pipeline
- Preprocessing is encapsulated in a scikit-learn `ColumnTransformer` fitted strictly on `X_train`:
  - **Amount**: `log1p(Amount)` transformation followed by `RobustScaler` to compress extreme right-tail skewness (\$0 to \$25,691).
  - **Time**: `RobustScaler` to normalize scale while dampening outlier distortion.
  - **V1–V28**: Retained as standardized PCA projections.

---

## Baseline Model Performance (Logistic Regression)

Trained with `class_weight='balanced'`, `C=1.0`, `solver='lbfgs'`, and `random_state=42`.

### Actual Validation Metrics (Threshold = 0.5)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **0.7552** | Primary ranking metric under class imbalance |
| **ROC-AUC** | **0.9691** | Excellent global discrimination across false-positive rates |
| **Recall** | **0.8772** | **50 out of 57 fraud cases** caught at initial threshold |
| **Precision** | **0.0254** | Low due to balanced class weighting shifting probabilities |
| **F1-Score** | **0.0494** | Uncalibrated baseline score |
| **True Negatives (TN)** | **54,988** | Non-fraud transactions correctly cleared |
| **False Positives (FP)** | **1,916** | Flagged for manual review under default threshold |
| **False Negatives (FN)** | **7** | Missed fraud cases |
| **True Positives (TP)** | **50** | Confirmed fraud cases detected |

> [!NOTE]
> Because `class_weight='balanced'` up-weights the minority fraud class by ~237x, the default threshold of 0.5 aggressively favors Recall (87.72%) at the cost of Precision (2.54%). In Day 4, cost-sensitive threshold tuning will recalibrate the decision boundary to minimize financial loss and drastically reduce False Positives.

---

## Limitations & Next Steps
1. **Linear Decision Boundary**: Logistic Regression cannot capture non-linear feature interactions (e.g. cross-correlations between V14, V17, and Amount).
2. **Precision vs Review Cost Tradeoff**: 1,916 false positives would overwhelm human fraud investigation teams.
3. **Upcoming Day 3 & 4 Deliverables**:
   - Engineer cyclical diurnal features ($\sin / \cos$ of hour-of-day).
   - Implement non-linear **XGBoost** classifier with tuned `scale_pos_weight`.
   - Perform cost-sensitive threshold calibration optimizing expected monetary savings.

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Complete Pipelines
```bash
# Day 1: Data Ingestion, Schema Validation, DuckDB SQL & EDA Visualizations
python src/run_day1_eda.py

# Day 2: Chronological Splitting, Pipeline Preprocessing & Baseline Model Training
python src/run_day2_pipeline.py
```

### 3. Run Test Suite
```bash
pytest tests/ -v
```
*(All 26 automated unit tests pass)*
