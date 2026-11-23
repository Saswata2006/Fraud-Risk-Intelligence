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
- Non-linear ensemble benchmarking: **Random Forest** vs. **XGBoost**.
- Optimized gradient boosting models (**XGBoost**) calibrated with cost-sensitive decision thresholds.
- Feature attribution and interpretability via **SHAP**.
- An analyst-facing **Streamlit** dashboard for transaction monitoring and real-time risk scoring.

---

## 7-Day Blueprint & Progress

| Day | Focus Area | Status | Deliverables |
| :--- | :--- | :--- | :--- |
| **Day 1** | **EDA, Ingestion & SQL Analytics** | **DONE** | Schema validation, DuckDB SQL analytics, 13 figures, data quality report, Jupyter EDA notebook. |
| **Day 2** | **Splitting, Feature Pipeline & Baseline** | **DONE** | Chronological 60/20/20 split, leakage audit, `log1p` + `RobustScaler` pipeline, Logistic Regression baseline. |
| **Day 3** | **Ensemble Modeling & Model Selection** | **DONE** | Random Forest (`balanced_subsample`), XGBoost (`scale_pos_weight`), chronological tuning, model comparison. |
| **Day 4** | **Cost-Sensitive Threshold Tuning & Calibration** | PENDING | Precision-recall tradeoff, expected financial loss optimization, probability calibration. |
| **Day 5** | **Explainability & Model Governance** | PENDING | SHAP TreeExplainer integration, waterfall/force plots, risk tiered scoring. |
| **Day 6** | **Interactive Streamlit Dashboard** | PENDING | Live transaction scoring UI, analyst review queue, interactive SHAP visualizer. |
| **Day 7** | **Hardening, Testing & Portfolio Delivery** | PENDING | End-to-end integration tests, benchmarking report, production documentation. |

---

## Architecture & Project Structure

```
fraud-risk-intelligence/
├── AGENTS.md                         # Multi-agent operating rules and guidelines
├── README.md                         # Project documentation and quickstart
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore specifications
├── data/
│   ├── raw/                          # Original data files (untracked, creditcard.csv)
│   ├── processed/                    # Feature-engineered Parquet partitions
│   └── database/                     # DuckDB analytical databases
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Day 1 EDA and SQL profiling
│   ├── 02_baseline_modeling.ipynb           # Day 2 Splitting and Baseline modeling
│   └── 03_tree_models_and_selection.ipynb   # Day 3 Tree Ensembles & Selection
├── reports/
│   ├── data_quality_report.md               # Data validation and audit report
│   └── figures/                             # Publication PNGs and interactive HTML plots
├── artifacts/
│   ├── baseline_logistic_regression.joblib  # Baseline LR pipeline
│   ├── baseline_metrics.json                # LR validation metrics
│   ├── random_forest_pipeline.joblib        # Random Forest pipeline
│   ├── xgboost_pipeline.joblib              # Tuned XGBoost pipeline
│   ├── model_comparison_metrics.json        # Comparative benchmark JSON
│   └── model_comparison_metrics.csv         # Comparative benchmark CSV
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── data_loader.py            # Schema validation and dataset loader
│   │   └── sql_analytics.py          # DuckDB vectorized SQL queries
│   ├── features/
│   │   ├── data_split.py             # Chronological splitting and duplicate audit
│   │   ├── preprocessor.py           # Leakage-safe ColumnTransformer pipelines
│   │   └── eda_analysis.py           # Statistical summaries and visualization engine
│   ├── models/
│   │   ├── baseline.py               # Baseline Logistic Regression pipeline
│   │   ├── random_forest.py          # Balanced Random Forest pipeline
│   │   ├── xgboost_model.py          # XGBoost pipeline & TimeSeriesSplit tuning
│   │   ├── model_comparison.py       # Benchmarking, PR curves & feature importance
│   │   └── evaluation.py             # Metrics calculation and diagnostic curves
│   ├── explainability/               # SHAP explainers, risk scoring engine (Day 5)
│   ├── app/                          # Streamlit monitoring and scoring UI (Day 6)
│   ├── run_day1_eda.py               # Day 1 execution script
│   ├── run_day2_pipeline.py          # Day 2 execution script
│   └── run_day3_pipeline.py          # Day 3 execution script
└── tests/
    ├── test_environment.py           # Environment and dependencies tests
    ├── test_data_validation.py       # Ingestion and schema validation tests
    ├── test_sql_analytics.py         # DuckDB SQL analytical tests
    ├── test_eda_visualizations.py    # Figure existence and non-empty tests
    ├── test_data_split.py            # Split bounds and cross-partition duplicate tests
    ├── test_baseline_model.py        # Baseline LR model and preprocessor tests
    └── test_tree_models.py           # RF, XGBoost, and comparison tests
```

---

## Validation Benchmark Results (Threshold = 0.5)

All models evaluated strictly on the **same held-out validation partition** ($N = 56,961$, 57 fraudulent transactions). Final test data ($N = 56,962$) remains untouched.

| Model | Average Precision (PR-AUC) | ROC-AUC | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Champion)** | **0.7613** | **0.9827** | **0.2176** | **0.8246** | **0.3443** | **47 / 57** | **169** | **10** |
| **Random Forest** | **0.7608** | **0.9630** | **0.9512** | **0.6842** | **0.7959** | **39 / 57** | **2** | **18** |
| **Logistic Regression** | **0.7552** | **0.9691** | **0.0254** | **0.8772** | **0.0494** | **50 / 57** | **1,916** | **7** |

---

## Why Average Precision (PR-AUC) is the Primary Metric
In an extreme class-imbalance setting ($0.173\%$ fraud):
- **Accuracy is misleading**: A trivial "always legitimate" model yields **$99.828\%$ accuracy** while missing **$100\%$ of fraud transactions**, leading to catastrophic financial loss.
- **ROC-AUC is over-optimistic**: Because the number of true negatives ($56,904$) dwarfs true positives ($57$), large influxes of false positives create only a tiny blip in the False Positive Rate ($\text{FPR} = \text{FP} / (\text{FP} + \text{TN})$), making models appear artificially near-perfect.
- **PR-AUC directly measures precision across recall levels**: Precision evaluates $\text{TP} / (\text{TP} + \text{FP})$, meaning false alarms immediately penalize the metric without dilution by true negatives.

---

## Model Selection Rationale

**XGBoost** is selected as the primary candidate model:
1. **Superior Global Ranking**: Achieves highest Average Precision (**$0.7613$**) and highest ROC-AUC (**$0.9827$**).
2. **Massive False Positive Reduction**: Reduces false positives by **over $91\%$** compared to the Logistic Regression baseline ($1,916 \to 169$) while maintaining high fraud detection recall ($82.46\%$, 47/57 cases caught).
3. **Non-Linear Interactions**: Seamlessly captures non-linear decision boundaries between anonymized PCA features (V14, V17, V12, V10) and financial transaction amounts.
4. **Foundation for Interpretability**: Natively supports exact, high-speed Shapley attribution via `TreeExplainer` in Day 5.

---

## Key Feature Importances
Both tree ensembles consistently identify similar core fraud signals without assigning invented business labels:
- **XGBoost (Gain Importance)**: **V14** ($0.258$), **V10** ($0.163$), **V17** ($0.142$), **V12** ($0.106$), **V4** ($0.081$).
- **Random Forest (Gini Impurity)**: **V17** ($0.169$), **V14** ($0.158$), **V12** ($0.144$), **V10** ($0.092$), **V11** ($0.063$).

---

## Limitations & Next Steps
1. **Uncalibrated Output Probabilities**: Raw prediction scores from tree ensembles trained with `scale_pos_weight` are shifted and do not reflect true posterior probabilities $P(\text{Fraud} \mid X)$.
2. **Fixed Threshold (0.5) is Suboptimal**: Financial institutions require cost-sensitive operating points based on cost of fraud vs cost of manual review.
3. **Upcoming Day 4 & Day 5 Deliverables**:
   - Probability calibration (Isotonic Regression / Platt Scaling).
   - Cost-sensitive threshold optimization to minimize expected monetary loss.
   - Comprehensive TreeSHAP explanations (global summary & local waterfall plots).

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run All Pipelines
```bash
# Day 1: EDA, Validation & DuckDB Analytics
python src/run_day1_eda.py

# Day 2: Splitting, Preprocessing & Logistic Regression Baseline
python src/run_day2_pipeline.py

# Day 3: Random Forest, XGBoost Benchmarking & Model Selection
python src/run_day3_pipeline.py
```

### 3. Run Complete Automated Test Suite
```bash
pytest tests/ -v
```
*(All 31 unit tests pass)*
