"""Day 4 Pipeline Runner: Calibration, Threshold Optimization, SHAP & Risk Scoring.

Execution flow:
  1.  Load dataset and recreate chronological splits.
  2.  Leakage audit — confirm split boundary times.
  3.  Carve calibration split from training partition (75% fit / 25% calib).
  4.  Re-train RF and XGBoost on fit subset (frozen Day-3 hyper-parameters).
  5.  Fit sigmoid calibrators on calib subset.
  6.  Evaluate raw vs calibrated: Brier score, log-loss on validation.
  7.  Plot reliability diagrams -> reports/figures/13_reliability_diagram.png
  8.  Threshold sweep on validation for both calibrated models.
  9.  Plot threshold curves -> reports/figures/14_threshold_curves.png
 10.  Select winning model + optimal threshold (objective: F1).
 11.  SHAP explanations on winning raw model.
 12.  Save artifacts (calibrated pipelines, metrics JSON, threshold JSON).
 13.  Print final summary.

Usage:
    source .venv/bin/activate
    python src/run_day4_pipeline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on the path when running from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.data_loader import load_raw_data as load_dataset
from src.features.data_split import create_chronological_splits
from src.models.calibration import (
    create_calibration_split,
    retrain_rf_on_fit,
    retrain_xgb_on_fit,
    fit_sigmoid_calibrator,
    build_calibrated_pipeline,
    compare_calibration_metrics,
    plot_reliability_diagrams,
    save_calibrated_pipeline,
)
from src.models.threshold_optimizer import (
    compute_threshold_metrics,
    find_optimal_threshold,
    precision_recall_at_k,
    plot_threshold_curves,
    save_threshold_metadata,
    FP_COST,
    FN_COST,
)
from src.explainability.shap_explainer import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_feature_importance,
    plot_shap_waterfall_local,
    get_expected_value,
)
from src.models.risk_scorer import predict_risk

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

# ===========================================================================
# STEP 1 — Load data and recreate splits
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 1: Loading dataset and recreating chronological splits")
print("=" * 65)

df = load_dataset(DATA_PATH)
train_df, val_df, test_df = create_chronological_splits(df)

TARGET = "Class"
FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_val = val_df[FEATURE_COLS]
y_val = val_df[TARGET]
X_test = test_df[FEATURE_COLS]  # HELD OUT — do not use for fitting or tuning

print(
    f"Train: {len(train_df):,} rows | fraud={int(y_train.sum())} ({y_train.mean()*100:.4f}%)\n"
    f"Val  : {len(val_df):,} rows | fraud={int(y_val.sum())} ({y_val.mean()*100:.4f}%)\n"
    f"Test : {len(test_df):,} rows | fraud={int(test_df[TARGET].sum())} "
    f"[STRICTLY HELD OUT — NOT USED IN DAY 4]"
)

# ===========================================================================
# STEP 2 — Leakage audit
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 2: Leakage audit")
print("=" * 65)

train_max_time = float(train_df["Time"].max())
val_min_time   = float(val_df["Time"].min())
val_max_time   = float(val_df["Time"].max())
test_min_time  = float(test_df["Time"].min())

assert train_max_time <= val_min_time, (
    f"LEAKAGE: train max Time ({train_max_time}) > val min Time ({val_min_time})"
)
assert val_max_time <= test_min_time, (
    f"LEAKAGE: val max Time ({val_max_time}) > test min Time ({test_min_time})"
)
print(
    f"[PASS] Train Time range : [0, {train_max_time:.0f}]\n"
    f"[PASS] Val   Time range : [{val_min_time:.0f}, {val_max_time:.0f}]\n"
    f"[PASS] Test  Time range : [{test_min_time:.0f}, ...]\n"
    "[PASS] No cross-partition time overlap — chronological split confirmed."
)

# ===========================================================================
# STEP 3 — Calibration split from training partition
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 3: Creating calibration split (75% fit / 25% calib)")
print("=" * 65)

fit_df, calib_df = create_calibration_split(train_df, fit_ratio=0.75)
X_fit   = fit_df[FEATURE_COLS]
y_fit   = fit_df[TARGET]
X_calib = calib_df[FEATURE_COLS]
y_calib = calib_df[TARGET]

# ===========================================================================
# STEP 4 — Re-train RF and XGBoost on fit subset (Day-3 frozen params)
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 4: Re-training models on fit subset (frozen Day-3 params)")
print("=" * 65)

print("[RF] Training Random Forest...")
rf_prep, rf_clf_raw = retrain_rf_on_fit(
    X_fit, y_fit,
    n_estimators=100, max_depth=10, min_samples_leaf=2, random_state=42,
)

print("[XGB] Training XGBoost...")
xgb_prep, xgb_clf_raw = retrain_xgb_on_fit(
    X_fit, y_fit,
    max_depth=3, learning_rate=0.1, n_estimators=100,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)

# ===========================================================================
# STEP 5 — Fit sigmoid calibrators on calib subset
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 5: Fitting sigmoid (Platt) calibrators on calib subset")
print("=" * 65)

X_calib_trans_rf  = rf_prep.transform(X_calib)
X_calib_trans_xgb = xgb_prep.transform(X_calib)

rf_cal_clf  = fit_sigmoid_calibrator(rf_clf_raw,  X_calib_trans_rf,  y_calib)
xgb_cal_clf = fit_sigmoid_calibrator(xgb_clf_raw, X_calib_trans_xgb, y_calib)

cal_rf_pipeline  = build_calibrated_pipeline(rf_prep,  rf_cal_clf)
cal_xgb_pipeline = build_calibrated_pipeline(xgb_prep, xgb_cal_clf)

# ===========================================================================
# STEP 6 — Calibration diagnostics (val only)
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 6: Calibration diagnostics (Brier score & log-loss on val)")
print("=" * 65)

rf_cal_metrics  = compare_calibration_metrics(rf_prep,  rf_clf_raw,  cal_rf_pipeline,  X_val, y_val, "RF")
xgb_cal_metrics = compare_calibration_metrics(xgb_prep, xgb_clf_raw, cal_xgb_pipeline, X_val, y_val, "XGBoost")

calibration_metrics = {
    "rf":  rf_cal_metrics,
    "xgb": xgb_cal_metrics,
}

# Save calibration metrics JSON
cal_metrics_path = ARTIFACTS_DIR / "calibration_metrics.json"
with open(cal_metrics_path, "w") as f:
    json.dump(calibration_metrics, f, indent=2)
print(f"[saved] Calibration metrics -> {cal_metrics_path}")

# ===========================================================================
# STEP 7 — Reliability diagrams
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 7: Plotting reliability diagrams")
print("=" * 65)

plot_reliability_diagrams(
    models={
        "RF Raw"      : (rf_prep,  rf_clf_raw,  False),
        "RF Calibrated": (None,    cal_rf_pipeline,  True),
        "XGB Raw"     : (xgb_prep, xgb_clf_raw, False),
        "XGB Calibrated": (None,   cal_xgb_pipeline, True),
    },
    X_val=X_val,
    y_val=y_val,
    output_dir=FIGURES_DIR,
)

# ===========================================================================
# STEP 8 — Threshold sweep on validation
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 8: Threshold sweep on validation set")
print("=" * 65)

thresholds = np.arange(0.05, 0.96, 0.025)
rf_thr_metrics  = compute_threshold_metrics(cal_rf_pipeline,  X_val, y_val, thresholds)
xgb_thr_metrics = compute_threshold_metrics(cal_xgb_pipeline, X_val, y_val, thresholds)

# Save threshold eval CSV
thr_csv_path = ARTIFACTS_DIR / "threshold_evaluation_metrics.csv"
rf_thr_metrics["model"]  = "Calibrated RF"
xgb_thr_metrics["model"] = "Calibrated XGBoost"
combined_thr = pd.concat([rf_thr_metrics, xgb_thr_metrics], ignore_index=True)
combined_thr.to_csv(thr_csv_path, index=False)
print(f"[saved] Threshold evaluation CSV -> {thr_csv_path}")
# Remove model column for per-model dicts
rf_thr_metrics  = rf_thr_metrics.drop(columns=["model"])
xgb_thr_metrics = xgb_thr_metrics.drop(columns=["model"])

# Precision@k / Recall@k at review budgets
print("\n--- Precision@k / Recall@k (review budget analysis) ---")
for k in [50, 100, 200]:
    print(f"  Calibrated RF  ", end="")
    precision_recall_at_k(cal_rf_pipeline,  X_val, y_val, k)
    print(f"  Calibrated XGB ", end="")
    precision_recall_at_k(cal_xgb_pipeline, X_val, y_val, k)

# ===========================================================================
# STEP 9 — Threshold curve plots
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 9: Plotting threshold curves")
print("=" * 65)

rf_opt_thr,  rf_best_row  = find_optimal_threshold(rf_thr_metrics,  objective="f1")
xgb_opt_thr, xgb_best_row = find_optimal_threshold(xgb_thr_metrics, objective="f1")

plot_threshold_curves(
    metrics_dfs={
        "Calibrated RF":      rf_thr_metrics,
        "Calibrated XGBoost": xgb_thr_metrics,
    },
    output_dir=FIGURES_DIR,
    optimal_thresholds={
        "Calibrated RF":      rf_opt_thr,
        "Calibrated XGBoost": xgb_opt_thr,
    },
)

# ===========================================================================
# STEP 10 — Model & threshold selection
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 10: Selecting winning model and threshold")
print("=" * 65)

rf_f1  = float(rf_best_row["f1"])
xgb_f1 = float(xgb_best_row["f1"])
print(f"Calibrated RF  F1 @ threshold={rf_opt_thr:.3f}  : {rf_f1:.4f}")
print(f"Calibrated XGB F1 @ threshold={xgb_opt_thr:.3f}  : {xgb_f1:.4f}")

if xgb_f1 >= rf_f1:
    winner_name     = "Calibrated XGBoost"
    winner_pipeline = cal_xgb_pipeline
    winner_raw_prep = xgb_prep
    winner_raw_clf  = xgb_clf_raw
    winner_thr      = xgb_opt_thr
    winner_row      = xgb_best_row
    model_type_str  = "xgboost"
else:
    winner_name     = "Calibrated RF"
    winner_pipeline = cal_rf_pipeline
    winner_raw_prep = rf_prep
    winner_raw_clf  = rf_clf_raw
    winner_thr      = rf_opt_thr
    winner_row      = rf_best_row
    model_type_str  = "random_forest"

print(f"\n[WINNER] {winner_name} | threshold={winner_thr:.3f}")

# ===========================================================================
# STEP 11 — SHAP explanations (on raw winner model, val set subsample)
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 11: Computing SHAP explanations")
print("=" * 65)

# Use all fraud rows + random sample of legitimate rows for SHAP
# Guarantees at least one fraud observation in the SHAP sample
val_fraud_idx = y_val[y_val == 1].index.tolist()
val_legit_idx = y_val[y_val == 0].index.tolist()
rng = np.random.default_rng(42)
n_legit_shap = min(1443, len(val_legit_idx))  # 1500 - len(fraud) cap
legit_sampled = rng.choice(val_legit_idx, size=n_legit_shap, replace=False).tolist()
shap_idx_all = sorted(val_fraud_idx + legit_sampled)

X_shap_sample = X_val.loc[shap_idx_all].reset_index(drop=True)
y_shap_sample = y_val.loc[shap_idx_all].reset_index(drop=True)

shap_values, X_trans_shap, feature_names = compute_shap_values(
    winner_raw_prep, winner_raw_clf, X_shap_sample, model_type=model_type_str
)

# Global summary and feature importance
plot_shap_summary(shap_values, X_trans_shap, feature_names, FIGURES_DIR, model_label=winner_name)
importance_df = plot_shap_feature_importance(shap_values, feature_names, FIGURES_DIR, model_label=winner_name)

print("\nTop 10 features by mean |SHAP|:")
print(importance_df.head(10).to_string(index=False))

# Expected value for waterfall plots
expected_value = get_expected_value(winner_raw_clf, model_type=model_type_str)

# Get calibrated probabilities for annotation
cal_probas = winner_pipeline.predict_proba(X_shap_sample)[:, 1]

# Find correctly classified fraud transaction (TP)
fraud_in_sample = np.where((y_shap_sample.values == 1) & (cal_probas >= winner_thr))[0]
legit_in_sample = np.where((y_shap_sample.values == 0) & (cal_probas < winner_thr))[0]

if len(fraud_in_sample) > 0:
    fraud_sample_idx = int(fraud_in_sample[0])
    plot_shap_waterfall_local(
        shap_values, X_trans_shap, feature_names, expected_value,
        sample_idx=fraud_sample_idx,
        output_dir=FIGURES_DIR,
        label="fraud",
        true_label=1,
        pred_proba=float(cal_probas[fraud_sample_idx]),
    )
else:
    print("[shap] No TP fraud found in SHAP subsample — skipping fraud waterfall.")

if len(legit_in_sample) > 0:
    legit_sample_idx = int(legit_in_sample[0])
    plot_shap_waterfall_local(
        shap_values, X_trans_shap, feature_names, expected_value,
        sample_idx=legit_sample_idx,
        output_dir=FIGURES_DIR,
        label="legit",
        true_label=0,
        pred_proba=float(cal_probas[legit_sample_idx]),
    )
else:
    print("[shap] No TN legit found in SHAP subsample — skipping legit waterfall.")

# ===========================================================================
# STEP 12 — Save artifacts
# ===========================================================================
print("\n" + "=" * 65)
print("STEP 12: Saving artifacts")
print("=" * 65)

save_calibrated_pipeline(cal_rf_pipeline,  ARTIFACTS_DIR / "calibrated_rf_pipeline.joblib")
save_calibrated_pipeline(cal_xgb_pipeline, ARTIFACTS_DIR / "calibrated_xgb_pipeline.joblib")

# Threshold metadata for the winner
save_threshold_metadata(
    threshold=winner_thr,
    model_name=winner_name,
    objective="f1",
    metrics_row=winner_row,
    fp_cost_assumption=FP_COST,
    fn_cost_assumption=FN_COST,
    filepath=ARTIFACTS_DIR / "threshold_metadata.json",
)

# Quick predict_risk smoke test on 5 val rows
print("\n[smoke test] predict_risk() on 5 validation rows:")
sample_5 = X_val.iloc[:5].copy()
risk_output = predict_risk(sample_5, winner_pipeline, threshold=winner_thr)
print(risk_output.to_string())

# ===========================================================================
# STEP 13 — Final summary
# ===========================================================================
print("\n" + "=" * 65)
print("DAY 4 COMPLETE — FINAL SUMMARY")
print("=" * 65)

print(f"""
Winning Model   : {winner_name}
Optimal Threshold: {winner_thr:.3f}  (objective: F1-score, validation set)
  Precision     : {float(winner_row['precision']):.4f}
  Recall        : {float(winner_row['recall']):.4f}
  F1-score      : {float(winner_row['f1']):.4f}
  False Positives: {int(winner_row['fp'])}
  False Negatives: {int(winner_row['fn'])}

Calibration (validation set):
  RF  — Brier raw={rf_cal_metrics['brier_raw']:.5f}  cal={rf_cal_metrics['brier_cal']:.5f}
         LogLoss raw={rf_cal_metrics['logloss_raw']:.5f}  cal={rf_cal_metrics['logloss_cal']:.5f}
  XGB — Brier raw={xgb_cal_metrics['brier_raw']:.5f}  cal={xgb_cal_metrics['brier_cal']:.5f}
         LogLoss raw={xgb_cal_metrics['logloss_raw']:.5f}  cal={xgb_cal_metrics['logloss_cal']:.5f}

Top SHAP features (raw model log-odds):
{importance_df.head(5).to_string(index=False)}

Artifacts:
  artifacts/calibrated_rf_pipeline.joblib
  artifacts/calibrated_xgb_pipeline.joblib
  artifacts/calibration_metrics.json
  artifacts/threshold_metadata.json
  artifacts/threshold_evaluation_metrics.csv
  reports/figures/13_reliability_diagram.png
  reports/figures/14_threshold_curves.png
  reports/figures/15_shap_summary.png
  reports/figures/16_shap_feature_importance.png
  reports/figures/17_shap_waterfall_fraud.png
  reports/figures/18_shap_waterfall_legit.png

NOTE: Test partition was NOT touched. Day 5 (Streamlit dashboard) next.
""")
