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
| **Day 5** | **Four-page Streamlit Dashboard** | **DONE** | Train/validation EDA, validation diagnostics, calibrated RF sample scoring with SHAP, CSV batch screening. |
| **Day 6** | **Final model audit and deployment readiness** | **DONE** | Frozen holdout evaluation and Community Cloud configuration. |
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
│   ├── app/                          # Four-page Streamlit analyst dashboard
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

All models evaluated strictly on the **same held-out validation partition** ($N = 56,961$, 57 fraudulent transactions). Final test data ($N = 56,962$) was reserved for the Day 6 evaluation.

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

### 1. Leakage-Safe Probability Calibration with FrozenEstimator
Raw tree models trained with `scale_pos_weight` (XGBoost) or `class_weight='balanced_subsample'` (Random Forest) produce uncalibrated margin scores that artificially inflate posterior fraud probabilities. 

To calibrate without target leakage:
- The training partition ($N = 170,884$) was partitioned chronologically into a **model-fit subset** ($75\%$, $N = 128,163$, 260 frauds) and a **disjoint calibration subset** ($25\%$, $N = 42,721$, 100 frauds).
- Both models were re-trained on the fit subset with frozen Day 3 hyperparameters.
- Sigmoid (Platt) calibrators were fitted using `CalibratedClassifierCV(estimator=FrozenEstimator(fitted_clf), method='sigmoid')` strictly on the calibration subset. Wrapping in `FrozenEstimator` strictly guarantees that the base tree ensemble is **never refitted** on calibration data.
- The final held-out test partition ($N = 56,962$) was excluded from fitting and selection.

**Validation Calibration Diagnostics ($N = 56,961$, 57 frauds)**:

| Model | Raw Brier Score | Calibrated Brier Score | Brier Improvement | Raw Log-Loss | Calibrated Log-Loss | Log-Loss Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Random Forest (Winner)** | 0.00076 | **0.00032** | **-57.5%** | 0.01259 | **0.00236** | **-81.3%** |
| **Calibrated XGBoost** | 0.00793 | **0.00071** | **-91.1%** | 0.03991 | **0.00348** | **-91.3%** |

*Artifacts*: `reports/figures/13_reliability_diagram.png/html`, `artifacts/calibration_metrics.json`.

---

### 2. Validation-Only Threshold Optimization & Cost-Sensitive Analysis
Evaluating models at an arbitrary $0.5$ threshold is financially suboptimal. An extended threshold sweep spanning **$0.005 \le \tau \le 0.95$** (specifically exploring sub-$0.05$ regions) was conducted on the **validation set only**.

**Illustrative Financial Cost Assumptions**:
- False Positive Cost ($\text{Cost}_{\text{FP}}$): **\$10** (manual analyst review + customer friction).
- False Negative Cost ($\text{Cost}_{\text{FN}}$): **\$500** (average unrecovered fraud loss per incident).

**Validation Optimal Operating Points (F1-Optimal vs. Cost-Optimal)**:

| Model | Strategy | Threshold ($\tau^*$) | Precision | Recall | F1-Score | TP | FP | FN | Financial Cost (\$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated RF (Selected)** | **F1-Optimal** | **0.390** | **0.9302** | **0.7018** | **0.8000** | **40 / 57** | **3** | **17** | **\$8,530** |
| **Calibrated RF** | **Cost-Optimal** | **0.005** | **0.3802** | **0.8070** | **0.5169** | **46 / 57** | **75** | **11** | **\$6,250** |
| **Calibrated XGBoost** | **F1-Optimal** | **0.500** | **0.9091** | **0.7018** | **0.7921** | **40 / 57** | **4** | **17** | **\$8,540** |
| **Calibrated XGBoost** | **Cost-Optimal** | **0.260** | **0.3239** | **0.8070** | **0.4623** | **46 / 57** | **96** | **11** | **\$6,460** |

**Review Budget Analysis ($\text{Precision}@k$ / $\text{Recall}@k$)**:
- **Budget $k = 50$ reviews**:
  - Calibrated RF: $\text{Precision}@50 = 86.0\%$, $\text{Recall}@50 = 75.44\%$ (43 / 57 caught).
  - Calibrated XGBoost: $\text{Precision}@50 = 82.0\%$, $\text{Recall}@50 = 71.93\%$ (41 / 57 caught).
- **Budget $k = 100$ reviews**:
  - Calibrated RF: $\text{Precision}@100 = 46.0\%$, $\text{Recall}@100 = 80.70\%$ (46 / 57 caught).
  - Calibrated XGBoost: $\text{Precision}@100 = 43.0\%$, $\text{Recall}@100 = 75.44\%$ (43 / 57 caught).
- **Budget $k = 200$ reviews**:
  - Calibrated RF: $\text{Precision}@200 = 23.0\%$, $\text{Recall}@200 = 80.70\%$ (46 / 57 caught).
  - Calibrated XGBoost: $\text{Precision}@200 = 23.0\%$, $\text{Recall}@200 = 80.70\%$ (46 / 57 caught).

**Model Selection Decision**:
**Calibrated Random Forest** is selected as the primary operational model:
- Achieves the highest F1-score (**$0.8000$** at $\tau = 0.390$).
- Lowest Brier score (**$0.00032$**) and lowest log-loss (**$0.00236$**).
- Under cost optimization ($\text{Cost}_{\text{FP}}=\$10, \text{Cost}_{\text{FN}}=\$500$), setting $\tau = 0.005$ captures **$80.70\%$ of all fraud** (46/57) with an expected financial loss of **\$6,250** (\$210 lower than XGBoost).
- Calibrated XGBoost remains a close finalist ($0.7921$ F1, \$6,460 cost-optimal).

*Artifacts*: `reports/figures/14_threshold_curves.png/html`, `artifacts/threshold_metadata.json`, `artifacts/threshold_evaluation_metrics.csv`.

---

### 3. Explainability via SHAP (TreeExplainer)
SHAP values were computed using `shap.TreeExplainer` on the raw tree model output.

> **Scientific Distinction**: For the Random Forest, TreeSHAP contributions explain the uncalibrated positive-class probability. For XGBoost, they explain the raw margin. Neither is the calibrated probability. Additivity ($\sum \phi_i = f(x) - E[f(x)]$) applies on the output scale being explained; the contributions are not additive percentage-point changes in calibrated probability.

- **Top Global Features (Mean |SHAP Value| on Calibrated RF)**:
  1. **V14** ($0.110$): Strongest global discriminator for fraud.
  2. **V12** ($0.077$): Negative excursions heavily increase fraud risk.
  3. **V4** ($0.052$): Positive values elevate fraud margin.
  4. **V11** ($0.047$): Secondary positive risk contributor.
  5. **V10** ($0.041$), **V17** ($0.032$), **V3** ($0.023$), **V16** ($0.018$).
- **Local Waterfall Attributions**:
  - **Fraud Case** ([`reports/figures/17_shap_waterfall_fraud.png`](reports/figures/17_shap_waterfall_fraud.png)): Negative shifts in V14 and positive shifts in V4 push the raw RF fraud output above baseline; the calibrated $P(\text{Fraud}) > 0.90$.
  - **Legitimate Case** ([`reports/figures/18_shap_waterfall_legit.png`](reports/figures/18_shap_waterfall_legit.png)): Central tendencies across key components hold output at base value, resulting in calibrated $P(\text{Fraud}) < 0.0006$.

*Artifacts*: `reports/figures/15_shap_summary.png`, `reports/figures/16_shap_feature_importance.png`, `reports/figures/17_shap_waterfall_fraud.png`, `reports/figures/18_shap_waterfall_legit.png`.

---

### 4. Risk Scoring Engine (`predict_risk`)
A production-ready scoring module (`src/models/risk_scorer.py`) applies the calibrated pipeline and segments transactions into actionable operational tiers:

```python
from src.models.calibration import load_calibrated_pipeline
from src.models.risk_scorer import predict_risk

pipeline = load_calibrated_pipeline("artifacts/calibrated_rf_pipeline.joblib")
scores_df = predict_risk(
    transactions_df,
    calibrated_pipeline=pipeline,
    threshold=0.390,
    low_cutoff=0.10,
    high_cutoff=0.50,
)
```

**Risk Tier Definitions**:
- **Low Risk** ($P < 0.10$): Low probability label.
- **Medium Risk** ($0.10 \le P < 0.50$): Intermediate probability label.
- **High Risk** ($P \ge 0.50$): High probability label.

These are descriptive tiers. The dashboard's review flag is a separate decision at the validation-selected RF threshold; it does not automatically approve, authenticate, or hold a transaction.

---

## Limitations & Next Steps
1. **Stationarity Assumption**: Chronological partitioning captures temporal ordering across the 2-day dataset window; long-term seasonal concept drift requires continuous retraining hooks.
2. **Final evaluation**: The final $20\%$ test partition ($N = 56,962$) was scored once on Day 6 after model and threshold selection.
3. **Remaining Deliverables**:
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

# Day 5 dashboard (uses saved models; does not train)
python -m streamlit run src/app/dashboard.py
```

### 3. Run Complete Automated Test Suite
```bash
pytest tests/ -v
```
Run the full suite after each change; the test count may grow as features are added.

### Day 5 analyst dashboard

Verified local launch from the repository root (no `PYTHONPATH` setting required):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run src/app/dashboard.py
```

Open the Local URL printed by Streamlit. If Matplotlib cannot write its default cache on your machine, prefix the second command with `MPLCONFIGDIR=/private/tmp`. This is a cache setting, not a Python import path.

The sidebar opens Data Intelligence, Model Performance, Fraud Prediction, and Batch Screening. The deployed app uses `data/demo/dashboard_summary.json`, `data/demo/validation_transactions_100.csv`, and the unchanged `artifacts/calibrated_rf_pipeline.joblib`. It does **not** need the ignored `data/raw/creditcard.csv` or the other saved model files at runtime. The summary contains train/validation EDA aggregates, validation metrics from saved Day 1–4 artifacts, and downsampled validation curve display points. Reported average precision values come from the full validation artifact, not from the downsampled display points. The final test partition is not accessed.

On **Batch Screening**, click **Download 100-row validation demo CSV**, then upload that file in **Transaction CSV**. Its 100 rows contain exactly the 30 numeric input features (`Time`, `Amount`, `V1`–`V28`) and no label. Ten fraud and 90 legitimate validation examples were deliberately sampled for demonstration; the sample is not representative of population prevalence. The scored download adds calibrated fraud probability, review decision, and risk tier. Optional identifier columns in other uploads are retained. Missing, nonnumeric, nonfinite, negative Time/Amount, duplicate, and output-name collision schemas are rejected.

On **Fraud Prediction**, choose a validation fraud or legitimate example and click **Explain this transaction** for a Plotly TreeSHAP chart. SHAP explains the uncalibrated RF output; the displayed fraud probability comes from the calibrated pipeline. **Review flag** means calibrated probability ≥ the validation-selected threshold. **Risk tier** is a separate label: Low < 0.10, Medium 0.10–< 0.50, High ≥ 0.50.

To regenerate the public demo bundle in a development checkout with the raw CSV and saved Day 1–4 artifacts, run `MPLCONFIGDIR=/private/tmp .venv/bin/python -m src.app.build_demo_data`. This reads only the chronological train and validation prefix and performs inference without fitting a model. The three runtime files above are explicitly exempted from `.gitignore`. All three are tracked on main. No deployment is performed by this command.

The review flag uses the RF threshold selected on validation (copied from `artifacts/threshold_metadata.json` into the compact summary); tier labels use the independent probability bands Low < 0.10, Medium 0.10–< 0.50, and High ≥ 0.50. All reported model metrics are validation-only. The uploaded CSV is scored as provided and is not used for evaluation or training.

## Day 6 final chronological evaluation

The saved model robust-scales Time, log-transforms Amount with log1p before robust scaling, and passes V1–V28 through. The Day 4 pipeline fits the preprocessor and RF on the first 128,163 training rows and fits sigmoid calibration on the following 42,721. Model selection and F1 threshold selection use the subsequent validation window. Earlier EDA, counts and duplicate audits included the full dataset. The repository records no use of test predictions for fitting or selection. It cannot establish what happened in unrecorded external runs.

The evaluator restored the existing chronological split and scored the last 56,962 transactions once using the saved calibrated pipeline and the persisted cutoff (displayed as 0.390). All test results use that probability vector. The report stores aggregates and the model SHA-256. The dashboard reads its original validation bundle. [Evaluation report](reports/final_test_metrics.json)

| Metric | Validation | Final test |
|---|---:|---:|
| Fraud / total | 57 / 56,961 | 75 / 56,962 |
| Average precision | 0.756553 | 0.791712 |
| ROC-AUC | 0.965722 | 0.966213 |
| Precision | 0.930233 | 0.898305 |
| Recall | 0.701754 | 0.706667 |
| F1 | 0.800000 | 0.791045 |
| Brier score | 0.000325 | 0.000419 |
| Log-loss | 0.002359 | 0.002875 |
| TN / FP / FN / TP | 56,901 / 3 / 17 / 40 | 56,881 / 6 / 22 / 53 |

Ranking improved by 0.0352 average precision. At the frozen operating point, precision fell by 3.19 percentage points and recall rose by 0.49 points. Both probability losses increased. The short observation window and 75 positive test cases limit conclusions about future fraud patterns.

### Streamlit Community Cloud

1. Publish the reviewed Day 6 changes to the deployment branch when ready. The required RF, demo CSV and summary JSON are already tracked on main.
2. Sign in to [Community Cloud](https://share.streamlit.io), select **Create app**, then **Yup, I have an app**.
3. Choose repository `Saswata2006/Fraud-Risk-Intelligence`, branch `main`, and main file `src/app/dashboard.py`.
4. Under **Advanced settings**, choose Python **3.12** and save. The app uses the root requirements.txt and .streamlit/config.toml. It needs no secrets.
5. Select **Deploy**, check the build logs, and use the bundled demo CSV to verify Batch Screening. [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
