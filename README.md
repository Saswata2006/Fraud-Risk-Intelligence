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
| **Day 4** | **Calibration, Threshold Tuning & SHAP** | **DONE** | Platt sigmoid calibration, Brier score/log loss audit, F1 threshold tuning, TreeSHAP, risk tiers. |
| **Day 5** | **Explainability & Model Governance** | PENDING | Model card, audit logs, compliance governance, drift monitoring hooks. |
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
│   ├── 03_tree_models_and_selection.ipynb   # Day 3 Tree Ensembles & Selection
│   └── 04_calibration_shap.ipynb            # Day 4 Calibration, Thresholds & SHAP
├── reports/
│   ├── data_quality_report.md               # Data validation and audit report
│   └── figures/                             # 18 publication PNGs and interactive HTML plots
├── artifacts/
│   ├── baseline_logistic_regression.joblib  # Baseline LR pipeline
│   ├── baseline_metrics.json                # LR validation metrics
│   ├── random_forest_pipeline.joblib        # Random Forest pipeline
│   ├── xgboost_pipeline.joblib              # Tuned XGBoost pipeline
│   ├── calibrated_rf_pipeline.joblib        # Calibrated Random Forest pipeline
│   ├── calibrated_xgb_pipeline.joblib       # Calibrated XGBoost pipeline
│   ├── calibration_metrics.json             # Raw vs Calibrated Brier & log-loss
│   ├── threshold_metadata.json              # Optimal operational threshold & metrics
│   ├── threshold_evaluation_metrics.csv     # Multi-threshold sweep results
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
│   │   ├── calibration.py            # Disjoint chronological calibration (Platt/sigmoid)
│   │   ├── threshold_optimizer.py    # Cost-sensitive threshold tuning & evaluation
│   │   ├── risk_scorer.py            # Multi-tier risk scoring engine (predict_risk)
│   │   └── evaluation.py             # Metrics calculation and diagnostic curves
│   ├── explainability/
│   │   └── shap_explainer.py         # TreeExplainer, summary, bar & waterfall plots
│   ├── app/                          # Streamlit monitoring and scoring UI (Day 6)
│   ├── run_day1_eda.py               # Day 1 execution script
│   ├── run_day2_pipeline.py          # Day 2 execution script
│   ├── run_day3_pipeline.py          # Day 3 execution script
│   └── run_day4_pipeline.py          # Day 4 execution script
└── tests/
    ├── test_environment.py           # Environment and dependencies tests
    ├── test_data_validation.py       # Ingestion and schema validation tests
    ├── test_sql_analytics.py         # DuckDB SQL analytical tests
    ├── test_eda_visualizations.py    # Figure existence and non-empty tests
    ├── test_data_split.py            # Split bounds and cross-partition duplicate tests
    ├── test_baseline_model.py        # Baseline LR model and preprocessor tests
    ├── test_tree_models.py           # RF, XGBoost, and comparison tests
    └── test_calibration_shap.py      # Calibration, thresholds, SHAP & risk scoring tests
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

---

## Day 4: Probability Calibration, Threshold Optimization & SHAP Explainability

### 1. Leakage-Safe Probability Calibration
Raw tree models trained with `scale_pos_weight` (XGBoost) or `class_weight='balanced_subsample'` (Random Forest) produce uncalibrated margin scores that artificially inflate posterior fraud probabilities. 

To calibrate without target leakage:
- The training partition ($N = 170,884$) was partitioned chronologically into a **model-fit subset** ($75\%$, $N = 128,163$, 260 frauds) and a **disjoint calibration subset** ($25\%$, $N = 42,721$, 100 frauds).
- Both models were re-trained on the fit subset with frozen Day 3 hyperparameters.
- Sigmoid (Platt) calibrators were fitted via `CalibratedClassifierCV(method='sigmoid', cv=None, ensemble=False)` strictly on the calibration subset.
- The final held-out test partition ($N = 56,962$) remained **completely untouched**.

**Validation Calibration Diagnostics ($N = 56,961$, 57 frauds)**:

| Model | Raw Brier Score | Calibrated Brier Score | Brier Improvement | Raw Log-Loss | Calibrated Log-Loss | Log-Loss Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Winner)** | 0.00793 | **0.00044** | **-94.4%** | 0.03991 | **0.00294** | **-92.6%** |
| **Random Forest** | 0.00076 | **0.00060** | **-21.1%** | 0.01259 | **0.00357** | **-71.6%** |

*Artifacts*: `reports/figures/13_reliability_diagram.png/html`, `artifacts/calibration_metrics.json`.

---

### 2. Validation-Only Threshold Optimization & Cost-Sensitive Analysis
Evaluating models at an arbitrary $0.5$ threshold is financially inefficient. A threshold sweep ($0.05 \le \tau \le 0.95$ with step $0.025$) was conducted on the **validation set only**.

**Illustrative Financial Cost Assumptions**:
- False Positive Cost ($\text{Cost}_{\text{FP}}$): **\$10** (manual analyst review + customer friction).
- False Negative Cost ($\text{Cost}_{\text{FN}}$): **\$500** (average unrecovered fraud loss per incident).

**Validation Optimal Operating Point (Objective: Maximize F1-Score)**:

| Model | Optimal Threshold ($\tau^*$) | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Financial Cost (\$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated XGBoost (Winner)** | **0.050** | **0.9545** | **0.7368** | **0.8317** | **42 / 57** | **2** | **15** | **\$7,520** |
| **Calibrated Random Forest** | **0.050** | **0.9459** | **0.6140** | **0.7447** | **35 / 57** | **2** | **22** | **\$11,020** |

**Review Budget Analysis ($\text{Precision}@k$ / $\text{Recall}@k$)**:
- **Budget $k = 50$ reviews**:
  - Calibrated XGBoost: $\text{Precision}@50 = 84.0\%$, $\text{Recall}@50 = 73.68\%$ (42 / 57 caught).
  - Calibrated RF: $\text{Precision}@50 = 84.0\%$, $\text{Recall}@50 = 73.68\%$ (42 / 57 caught).
- **Budget $k = 100$ reviews**:
  - Calibrated XGBoost: $\text{Precision}@100 = 42.0\%$, $\text{Recall}@100 = 73.68\%$.
  - Calibrated RF: $\text{Precision}@100 = 42.0\%$, $\text{Recall}@100 = 73.68\%$.
- **Budget $k = 200$ reviews**:
  - Calibrated XGBoost: $\text{Precision}@200 = 21.5\%$, $\text{Recall}@200 = 75.44\%$ (43 / 57 caught).
  - Calibrated RF: $\text{Precision}@200 = 22.5\%$, $\text{Recall}@200 = 78.95\%$ (45 / 57 caught).

**Model Selection Decision**:
**Calibrated XGBoost** is selected as the production model. At $\tau^* = 0.050$, it delivers an **F1-score of 0.8317** (+0.087 over RF), detects **42 out of 57 fraud cases** (7 more than RF) with only **2 false positives**, and saves **\$3,500** in illustrative financial loss compared to Random Forest.

*Artifacts*: `reports/figures/14_threshold_curves.png/html`, `artifacts/threshold_metadata.json`, `artifacts/threshold_evaluation_metrics.csv`.

---

### 3. Explainability via SHAP (TreeExplainer)
SHAP values were computed using `shap.TreeExplainer` on the raw tree model's log-odds margin space. 

> **Scientific Distinction**: SHAP explains raw model margin scores (pre-calibration log-odds), NOT calibrated probabilities. Because Platt calibration applies a monotonic sigmoid transformation, feature ranking and directional attribution are strictly preserved.

- **Top 10 Global Features (Mean |SHAP Value|)**:
  1. **V14** ($1.519$): Dominant negative indicator of fraud.
  2. **V4** ($1.080$): Strong positive indicator pushing risk upward.
  3. **V12** ($0.970$): Major risk suppressor when values are positive.
  4. **V10** ($0.615$): High values heavily discount fraud probability.
  5. **V8** ($0.329$): Bimodal contribution across high and low percentiles.
  6. **V22** ($0.250$), **V1** ($0.245$), **V25** ($0.229$), **V11** ($0.225$), **V19** ($0.224$).
- **Local Explanations**:
  - Correctly flagged fraud transactions exhibit severe negative excursions in **V14** and elevated values in **V4**, driving calibrated fraud risk $> 0.90$.
  - Legitimate transactions maintain standard PCA profiles with expected base values ($E[f(x)] = -5.86$), holding calibrated probability $< 0.001$.

*Artifacts*: `reports/figures/15_shap_summary.png`, `reports/figures/16_shap_feature_importance.png`, `reports/figures/17_shap_waterfall_fraud.png`, `reports/figures/18_shap_waterfall_legit.png`.

---

### 4. Risk Scoring Engine (`predict_risk`)
A production-ready scoring module (`src/models/risk_scorer.py`) applies the calibrated pipeline and segments transactions into actionable operational tiers:

```python
from src.models.calibration import load_calibrated_pipeline
from src.models.risk_scorer import predict_risk

pipeline = load_calibrated_pipeline("artifacts/calibrated_xgb_pipeline.joblib")
scores_df = predict_risk(
    transactions_df,
    calibrated_pipeline=pipeline,
    threshold=0.050,
    low_cutoff=0.10,
    high_cutoff=0.50,
)
```

**Risk Tier Definitions**:
- **Low Risk** ($P < 0.10$): Automated approval; no human intervention needed.
- **Medium Risk** ($0.10 \le P < 0.50$): Step-up authentication (SMS OTP, biometric prompt).
- **High Risk** ($P \ge 0.50$): Immediate temporary hold and routing to fraud analyst queue.

---

## Limitations & Next Steps
1. **Stationarity Assumption**: Chronological partitioning captures temporal ordering across the 2-day dataset window; long-term seasonal concept drift requires continuous retraining hooks.
2. **Held-Out Test Set Untouched**: The final $20\%$ test partition ($N = 56,962$) has remained strictly held out and will be evaluated once in Day 7 for portfolio-grade reporting.
3. **Upcoming Day 5–7 Deliverables**:
   - **Day 5**: Model governance, audit logs, and compliance documentation.
   - **Day 6**: Interactive Streamlit analyst dashboard with real-time SHAP visualizer.
   - **Day 7**: End-to-end integration tests, benchmarking report, and final portfolio delivery.

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

# Day 4: Calibration, Threshold Optimization & SHAP Explainability
python src/run_day4_pipeline.py

# Generate Day 4 Jupyter Notebook
python src/generate_day4_notebook.py
```

### 3. Run Complete Automated Test Suite
```bash
pytest tests/ -v
```
*(All 53 unit tests pass)*
