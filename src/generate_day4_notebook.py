"""Generate Day 4 Jupyter notebook: 04_calibration_shap.ipynb.

This script produces a reproducible notebook documenting:
- Calibration approach and methodology
- Raw vs calibrated Brier score and log-loss comparison
- Threshold sweep and optimal selection
- SHAP global and local explanations
- Risk scoring with predict_risk()

Usage:
    source .venv/bin/activate
    python src/generate_day4_notebook.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.threshold_optimizer import FP_COST, FN_COST

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_calibration_shap.ipynb"

CELLS = [
    # ---- Markdown: Header ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Day 4: Probability Calibration, Threshold Optimization & SHAP Explainability\n",
            "\n",
            "**Project**: Explainable Credit Card Fraud Detection and Risk Scoring  \n",
            "**Stage**: Day 4 — Leakage-safe calibration, threshold selection, SHAP, risk scoring  \n",
            "**Dataset**: Kaggle ULB Credit Card Fraud Detection (284,807 transactions)  \n",
            "\n",
            "## Overview\n",
            "1. Reload chronological splits\n",
            "2. Carve calibration subset from training partition (75/25)\n",
            "3. Retrain RF & XGBoost on fit subset (Day-3 frozen hyperparameters)\n",
            "4. Fit sigmoid (Platt) calibrators on calibration subset\n",
            "5. Compare raw vs calibrated Brier score and log-loss\n",
            "6. Threshold sweep → optimal F1 threshold\n",
            "7. SHAP global summary + feature importance\n",
            "8. SHAP local waterfall plots (fraud & legitimate)\n",
            "9. Risk scoring with `predict_risk()`\n",
        ],
    },
    # ---- Code: Imports ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import json\n",
            "\n",
            "PROJECT_ROOT = Path.cwd().parent\n",
            "sys.path.insert(0, str(PROJECT_ROOT))\n",
            "\n",
            "from src.ingestion.data_loader import load_raw_data\n",
            "from src.features.data_split import create_chronological_splits\n",
            "from src.models.calibration import (\n",
            "    create_calibration_split, retrain_rf_on_fit, retrain_xgb_on_fit,\n",
            "    fit_sigmoid_calibrator, build_calibrated_pipeline, compare_calibration_metrics,\n",
            "    plot_reliability_diagrams, save_calibrated_pipeline\n",
            ")\n",
            "from src.models.threshold_optimizer import (\n",
            "    compute_threshold_metrics, find_optimal_threshold,\n",
            "    compare_f1_and_cost_thresholds, precision_recall_at_k,\n",
            "    plot_threshold_curves, DEFAULT_THRESHOLDS, FP_COST, FN_COST\n",
            ")\n",
            "from src.explainability.shap_explainer import (\n",
            "    compute_shap_values, plot_shap_summary, plot_shap_feature_importance,\n",
            "    plot_shap_waterfall_local, get_expected_value\n",
            ")\n",
            "from src.models.risk_scorer import predict_risk\n",
            "\n",
            "DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'creditcard.csv'\n",
            "FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'\n",
            "ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'\n",
            "print('Imports OK')\n",
        ],
    },
    # ---- Markdown: Step 1 ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Step 1: Load Data and Recreate Splits\n"],
    },
    # ---- Code: Load splits ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = load_raw_data(DATA_PATH)\n",
            "train_df, val_df, test_df = create_chronological_splits(df)\n",
            "\n",
            "TARGET = 'Class'\n",
            "FEATURE_COLS = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]\n",
            "\n",
            "X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]\n",
            "X_val, y_val     = val_df[FEATURE_COLS],   val_df[TARGET]\n",
            "X_test           = test_df[FEATURE_COLS]   # HELD OUT\n",
            "\n",
            "print(f'Train: {len(train_df):,} rows | fraud={y_train.sum()}')\n",
            "print(f'Val  : {len(val_df):,} rows | fraud={y_val.sum()}')\n",
            "print(f'Test : {len(test_df):,} rows [STRICTLY HELD OUT]')\n",
        ],
    },
    # ---- Markdown: Calibration ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2–5: Leakage-Safe Calibration\n",
            "\n",
            "**Approach**:\n",
            "- Split training partition: first 75% → model fitting, last 25% → calibrator fitting.\n",
            "- Re-train RF and XGBoost with **frozen Day-3 hyperparameters** on the fit subset.\n",
            "- Fit `CalibratedClassifierCV(method='sigmoid')` with `FrozenEstimator` on calibration subset.\n",
            "- Guarantees zero re-fitting of base estimator trees on calibration data.\n",
            "- Validation data is **never** used for fitting or calibration.\n",
        ],
    },
    # ---- Code: Calibration ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fit_df, calib_df = create_calibration_split(train_df, fit_ratio=0.75)\n",
            "X_fit, y_fit = fit_df[FEATURE_COLS], fit_df[TARGET]\n",
            "X_calib, y_calib = calib_df[FEATURE_COLS], calib_df[TARGET]\n",
            "\n",
            "# Retrain on fit subset\n",
            "rf_prep, rf_clf_raw = retrain_rf_on_fit(X_fit, y_fit, n_estimators=100, max_depth=10)\n",
            "xgb_prep, xgb_clf_raw = retrain_xgb_on_fit(X_fit, y_fit)\n",
            "\n",
            "# Fit calibrators\n",
            "rf_cal_clf  = fit_sigmoid_calibrator(rf_clf_raw,  rf_prep.transform(X_calib),  y_calib)\n",
            "xgb_cal_clf = fit_sigmoid_calibrator(xgb_clf_raw, xgb_prep.transform(X_calib), y_calib)\n",
            "\n",
            "cal_rf_pipeline  = build_calibrated_pipeline(rf_prep,  rf_cal_clf)\n",
            "cal_xgb_pipeline = build_calibrated_pipeline(xgb_prep, xgb_cal_clf)\n",
            "\n",
            "rf_metrics  = compare_calibration_metrics(rf_prep,  rf_clf_raw,  cal_rf_pipeline,  X_val, y_val, 'RF')\n",
            "xgb_metrics = compare_calibration_metrics(xgb_prep, xgb_clf_raw, cal_xgb_pipeline, X_val, y_val, 'XGB')\n",
            "\n",
            "print('\\nCalibration diagnostics (validation set):')\n",
            "for name, m in [('RF', rf_metrics), ('XGBoost', xgb_metrics)]:\n",
            "    print(f'  {name}: Brier raw={m[\"brier_raw\"]:.5f} → cal={m[\"brier_cal\"]:.5f} | '\n",
            "          f'LogLoss raw={m[\"logloss_raw\"]:.5f} → cal={m[\"logloss_cal\"]:.5f}')\n",
        ],
    },
    # ---- Code: Reliability diagrams ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load saved figure\n",
            "from IPython.display import Image\n",
            "Image(str(FIGURES_DIR / '13_reliability_diagram.png'), width=700)\n",
        ],
    },
    # ---- Markdown: Threshold ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Threshold Optimization (Validation Set)\n",
            "\n",
            "**Cost assumptions (illustrative only — not real business data)**:\n",
            f"- FP cost = \\${FP_COST:.0f} (analyst review + customer friction)\n",
            f"- FN cost = \\${FN_COST:.0f} (assumed average fraud loss)\n",
        ],
    },
    # ---- Code: Threshold sweep ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "thresholds = DEFAULT_THRESHOLDS\n",
            "rf_thr_df  = compute_threshold_metrics(cal_rf_pipeline,  X_val, y_val, thresholds)\n",
            "xgb_thr_df = compute_threshold_metrics(cal_xgb_pipeline, X_val, y_val, thresholds)\n",
            "\n",
            "(rf_f1_thr, rf_f1_row), (rf_cost_thr, rf_cost_row) = compare_f1_and_cost_thresholds(rf_thr_df)\n",
            "(xgb_f1_thr, xgb_f1_row), (xgb_cost_thr, xgb_cost_row) = compare_f1_and_cost_thresholds(xgb_thr_df)\n",
            "\n",
            "print('Calibrated RF:')\n",
            "print(f'  F1-optimal   : threshold={rf_f1_thr:.3f} | F1={rf_f1_row[\"f1\"]:.4f} P={rf_f1_row[\"precision\"]:.4f} R={rf_f1_row[\"recall\"]:.4f} Cost=\\${rf_f1_row[\"financial_cost\"]:.0f}')\n",
            "print(f'  Cost-optimal : threshold={rf_cost_thr:.3f} | Cost=\\${rf_cost_row[\"financial_cost\"]:.0f} F1={rf_cost_row[\"f1\"]:.4f} P={rf_cost_row[\"precision\"]:.4f} R={rf_cost_row[\"recall\"]:.4f}')\n",
            "print('\\nCalibrated XGBoost:')\n",
            "print(f'  F1-optimal   : threshold={xgb_f1_thr:.3f} | F1={xgb_f1_row[\"f1\"]:.4f} P={xgb_f1_row[\"precision\"]:.4f} R={xgb_f1_row[\"recall\"]:.4f} Cost=\\${xgb_f1_row[\"financial_cost\"]:.0f}')\n",
            "print(f'  Cost-optimal : threshold={xgb_cost_thr:.3f} | Cost=\\${xgb_cost_row[\"financial_cost\"]:.0f} F1={xgb_cost_row[\"f1\"]:.4f} P={xgb_cost_row[\"precision\"]:.4f} R={xgb_cost_row[\"recall\"]:.4f}')\n",
        ],
    },
    # ---- Code: Threshold plot ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["Image(str(FIGURES_DIR / '14_threshold_curves.png'), width=900)\n"],
    },
    # ---- Markdown: SHAP ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7–8: SHAP Explainability\n",
            "\n",
            "> **Important**: SHAP values explain the **raw tree model's decision margin (log-odds output)**,  \n",
            "> NOT the sigmoid-calibrated probabilities. Platt calibration applies a monotonic sigmoid mapping,  \n",
            "> but because sigmoid scaling is non-linear and saturates in the tails, feature additivity  \n",
            "> holds strictly in margin space and does not equal additive probability percentages.\n",
        ],
    },
    # ---- Code: SHAP ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load saved SHAP figures\n",
            "print('SHAP Summary Plot:')\n",
            "display(Image(str(FIGURES_DIR / '15_shap_summary.png'), width=800))\n",
            "print('SHAP Feature Importance:')\n",
            "display(Image(str(FIGURES_DIR / '16_shap_feature_importance.png'), width=700))\n",
        ],
    },
    # ---- Code: Waterfall ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Waterfall — Fraud transaction:')\n",
            "display(Image(str(FIGURES_DIR / '17_shap_waterfall_fraud.png'), width=750))\n",
            "print('Waterfall — Legitimate transaction:')\n",
            "display(Image(str(FIGURES_DIR / '18_shap_waterfall_legit.png'), width=750))\n",
        ],
    },
    # ---- Markdown: Risk Scoring ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 9: Risk Scoring with `predict_risk()`\n",
            "\n",
            "Risk tiers:\n",
            "- **Low**: fraud_probability < 0.10\n",
            "- **Medium**: 0.10 ≤ fraud_probability < 0.50\n",
            "- **High**: fraud_probability ≥ 0.50\n",
        ],
    },
    # ---- Code: Risk scoring demo ----
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load saved calibrated pipeline and threshold\n",
            "from src.models.calibration import load_calibrated_pipeline\n",
            "from src.models.threshold_optimizer import load_threshold_metadata\n",
            "\n",
            "metadata = load_threshold_metadata(ARTIFACTS_DIR / 'threshold_metadata.json')\n",
            "winning_model = metadata['model_name']\n",
            "opt_thr = metadata['optimal_threshold']\n",
            "\n",
            "if 'XGBoost' in winning_model:\n",
            "    pipeline = load_calibrated_pipeline(ARTIFACTS_DIR / 'calibrated_xgb_pipeline.joblib')\n",
            "else:\n",
            "    pipeline = load_calibrated_pipeline(ARTIFACTS_DIR / 'calibrated_rf_pipeline.joblib')\n",
            "\n",
            "print(f'Winning model: {winning_model}')\n",
            "print(f'Optimal threshold: {opt_thr:.3f} (objective: F1-score, validation set only)')\n",
            "\n",
            "# Score 10 validation rows\n",
            "sample = X_val.iloc[:10].copy()\n",
            "risk_df = predict_risk(sample, pipeline, threshold=opt_thr)\n",
            "display(risk_df)\n",
        ],
    },
    # ---- Markdown: Conclusion ----
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Summary\n",
            "\n",
            "| Item | Value |\n",
            "|------|-------|\n",
            "| Calibration method | Platt sigmoid (`CalibratedClassifierCV`, `cv='prefit'`) |\n",
            "| Calibration subset | Last 25% of training partition (chronological, disjoint) |\n",
            "| Test partition | **NEVER TOUCHED** |\n",
            "| Threshold selection | F1-optimal on validation set only |\n",
            "| SHAP explainer | `shap.TreeExplainer` on raw classifier (not calibrated wrapper) |\n",
            "| SHAP scope | Explains raw log-odds, NOT calibrated probabilities |\n",
            "\n",
            "**Next**: Day 5 — Streamlit interactive dashboard.\n",
        ],
    },
]


def build_notebook(cells):
    """Assemble a nbformat-compatible notebook dict."""
    nb_cells = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cell["source"],
            })
        else:
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell["source"],
            })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "cells": nb_cells,
    }


if __name__ == "__main__":
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(CELLS)
    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Notebook saved -> {NOTEBOOK_PATH}")
