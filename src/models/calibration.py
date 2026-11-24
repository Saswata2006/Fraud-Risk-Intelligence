"""Probability calibration module for fraud detection models.

Provides leakage-safe sigmoid (Platt) calibration using a disjoint chronological
calibration subset carved from the training partition.  The workflow:

  1. Split the training DataFrame chronologically: first ``fit_ratio`` fraction
     for model training, remaining fraction for calibrator fitting.
  2. Re-train (or accept) a base classifier on the fit subset only.
  3. Wrap the fitted classifier in ``FrozenEstimator`` and fit
     ``CalibratedClassifierCV(method='sigmoid')`` on the calibration subset.
  4. Build a full sklearn Pipeline:  preprocessor -> calibrated_classifier.

SHAP NOTE: SHAP explanations must be computed on the *raw* (uncalibrated)
classifier, not the CalibratedClassifierCV wrapper, because tree shap
operates on the underlying tree structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

import xgboost as xgb

from src.features.preprocessor import create_feature_preprocessor
from src.models.xgboost_model import compute_scale_pos_weight


# ---------------------------------------------------------------------------
# Calibration split
# ---------------------------------------------------------------------------

def create_calibration_split(
    train_df: pd.DataFrame,
    fit_ratio: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training partition into model-fit and calibration subsets.

    The first floor(len(train_df) * fit_ratio) positions form the fit subset.
    Later positions form the calibration subset. Tied elapsed times may span
    the boundary. Index labels can repeat because slicing uses row positions.

    Args:
        train_df: Full training partition DataFrame (must already be
            chronologically ordered, as produced by
            ``create_chronological_splits``).
        fit_ratio: Fraction of training rows to use for model fitting.
            Default 0.75 (75 / 25 split).

    Returns:
        Tuple of (fit_df, calib_df) DataFrames.
    """
    if not (0.0 < fit_ratio < 1.0):
        raise ValueError(f"fit_ratio must be in (0, 1), got {fit_ratio}")
    if len(train_df) < 2:
        raise ValueError("At least two training rows are required for calibration splitting.")
    if "Time" not in train_df:
        raise ValueError("Training rows require a Time column in elapsed seconds.")
    time = train_df["Time"]
    if time.dtype.kind not in "ifu" or time.isna().any() or not np.isfinite(time).all():
        raise ValueError("Time must contain finite numeric elapsed seconds.")
    if not time.is_monotonic_increasing:
        raise ValueError("Training rows must be in chronological order by Time.")

    n_fit = int(len(train_df) * fit_ratio)
    if not (0 < n_fit < len(train_df)):
        raise ValueError("fit_ratio must leave rows in both fit and calibration subsets.")
    fit_df = train_df.iloc[:n_fit].copy()
    calib_df = train_df.iloc[n_fit:].copy()

    print(
        f"[calibration] Calibration split:"
        f"  fit={len(fit_df):,} rows (fraud={int(fit_df['Class'].sum())})"
        f" | calib={len(calib_df):,} rows (fraud={int(calib_df['Class'].sum())})"
    )
    return fit_df, calib_df


# ---------------------------------------------------------------------------
# Model re-training on fit subset (frozen Day-3 hyper-parameters)
# ---------------------------------------------------------------------------

def retrain_rf_on_fit(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 2,
    random_state: int = 42,
) -> Tuple[Any, Any]:
    """Re-train Random Forest with frozen Day-3 hyper-parameters on X_fit.

    Returns the fitted preprocessor and fitted classifier separately so the
    calibration step can be applied to the classifier after transforming
    X_calib through the preprocessor.

    Args:
        X_fit: Feature DataFrame for model fitting.
        y_fit: Target Series for model fitting.
        n_estimators: Number of trees (Day-3 frozen value: 100).
        max_depth: Maximum tree depth (Day-3 frozen value: 10).
        min_samples_leaf: Minimum leaf samples (Day-3 frozen value: 2).
        random_state: Reproducibility seed.

    Returns:
        Tuple of (fitted_preprocessor, fitted_classifier).
    """
    preprocessor = create_feature_preprocessor()
    X_fit_trans = preprocessor.fit_transform(X_fit)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_fit_trans, y_fit)
    print(f"[calibration] RF re-trained on {len(X_fit):,} samples (fit subset).")
    return preprocessor, clf


def retrain_xgb_on_fit(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> Tuple[Any, Any]:
    """Re-train XGBoost with frozen Day-3 hyper-parameters on X_fit.

    scale_pos_weight is computed from y_fit only (no leakage).

    Args:
        X_fit: Feature DataFrame for model fitting.
        y_fit: Target Series for model fitting.
        max_depth: Day-3 frozen value: 3.
        learning_rate: Day-3 frozen value: 0.1.
        n_estimators: Day-3 frozen value: 100.
        subsample: Day-3 frozen value: 0.8.
        colsample_bytree: Day-3 frozen value: 0.8.
        random_state: Reproducibility seed.

    Returns:
        Tuple of (fitted_preprocessor, fitted_classifier).
    """
    spw = compute_scale_pos_weight(y_fit)
    preprocessor = create_feature_preprocessor()
    X_fit_trans = preprocessor.fit_transform(X_fit)

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_fit_trans, y_fit)
    print(
        f"[calibration] XGBoost re-trained on {len(X_fit):,} samples"
        f" (scale_pos_weight={spw:.2f})."
    )
    return preprocessor, clf


# ---------------------------------------------------------------------------
# Sigmoid calibration
# ---------------------------------------------------------------------------

def fit_sigmoid_calibrator(
    fitted_clf: Any,
    X_calib_trans: np.ndarray,
    y_calib: pd.Series,
) -> CalibratedClassifierCV:
    """Wrap a pre-fitted classifier with Platt (sigmoid) calibration.

    Uses ``FrozenEstimator(fitted_clf)`` to strictly guarantee that the
    base classifier is NOT re-fitted during calibration. Only the sigmoid
    parameters are estimated on (X_calib_trans, y_calib).

    Args:
        fitted_clf: Already-trained classifier (RF or XGB).
        X_calib_trans: Transformed feature matrix for calibrator fitting.
        y_calib: Target labels for calibrator fitting.

    Returns:
        Fitted CalibratedClassifierCV instance.
    """
    frozen_clf = FrozenEstimator(fitted_clf)
    calibrated_clf = CalibratedClassifierCV(
        estimator=frozen_clf,
        method="sigmoid",
    )
    calibrated_clf.fit(X_calib_trans, y_calib)
    print(f"[calibration] Sigmoid calibrator fitted on {len(y_calib):,} samples using FrozenEstimator.")
    return calibrated_clf


def build_calibrated_pipeline(
    fitted_preprocessor: Any,
    calibrated_clf: CalibratedClassifierCV,
) -> Pipeline:
    """Assemble a full predict-ready Pipeline: preprocessor + calibrated_clf.

    The preprocessor is already fitted; it will only transform at inference.

    Args:
        fitted_preprocessor: Fitted ColumnTransformer.
        calibrated_clf: Fitted CalibratedClassifierCV.

    Returns:
        Assembled Pipeline (preprocessor already fitted; call .predict_proba
        directly on raw feature DataFrames).
    """
    pipeline = Pipeline([
        ("preprocessor", fitted_preprocessor),
        ("classifier", calibrated_clf),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def compare_calibration_metrics(
    raw_preprocessor: Any,
    raw_clf: Any,
    cal_pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "model",
) -> Dict[str, float]:
    """Compare raw vs calibrated model using Brier score and log loss on validation data.

    Args:
        raw_preprocessor: Fitted preprocessor for the raw model.
        raw_clf: Fitted (uncalibrated) classifier.
        cal_pipeline: Fitted calibrated pipeline (preprocessor + calibrated clf).
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
        model_name: Label for logging.

    Returns:
        Dict with keys: brier_raw, brier_cal, logloss_raw, logloss_cal.
    """
    X_val_trans = raw_preprocessor.transform(X_val)
    raw_proba = raw_clf.predict_proba(X_val_trans)[:, 1]
    cal_proba = cal_pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "brier_raw": float(brier_score_loss(y_val, raw_proba)),
        "brier_cal": float(brier_score_loss(y_val, cal_proba)),
        "logloss_raw": float(log_loss(y_val, raw_proba)),
        "logloss_cal": float(log_loss(y_val, cal_proba)),
    }

    print(
        f"[calibration] {model_name} | "
        f"Brier raw={metrics['brier_raw']:.5f} -> cal={metrics['brier_cal']:.5f} | "
        f"LogLoss raw={metrics['logloss_raw']:.5f} -> cal={metrics['logloss_cal']:.5f}"
    )
    return metrics


def plot_reliability_diagrams(
    models: Dict[str, Tuple[Any, Any, bool]],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Path,
    n_bins: int = 10,
) -> None:
    """Plot reliability (calibration) diagrams for multiple models.

    Each entry in ``models`` is: label -> (preprocessor, clf, is_pipeline).
    If is_pipeline=True, clf is a full Pipeline and preprocessor is ignored.

    Args:
        models: Dict mapping label string to (preprocessor, clf, is_pipeline).
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
        output_dir: Directory to save figures.
        n_bins: Number of probability bins for the reliability curve.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for label, (preprocessor, clf, is_pipeline) in models.items():
        if is_pipeline:
            proba = clf.predict_proba(X_val)[:, 1]
        else:
            X_trans = preprocessor.transform(X_val)
            proba = clf.predict_proba(X_trans)[:, 1]

        disp = CalibrationDisplay.from_predictions(
            y_val, proba, n_bins=n_bins, name=label, ax=ax
        )

    ax.set_title("Reliability Diagrams (Calibration Curves)", fontsize=13)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Fraud Rate)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    png_path = output_dir / "13_reliability_diagram.png"
    html_path = output_dir / "13_reliability_diagram.html"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    # Simple HTML wrapper
    html_content = f"""<!DOCTYPE html><html><body>
<h2>Reliability Diagrams</h2>
<img src="{png_path.name}" style="max-width:800px;">
</body></html>"""
    html_path.write_text(html_content)
    plt.close(fig)
    print(f"[calibration] Reliability diagram saved -> {png_path}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_calibrated_pipeline(pipeline: Pipeline, filepath: Path) -> None:
    """Serialize a calibrated pipeline to disk with joblib.

    Args:
        pipeline: Fitted Pipeline (preprocessor + calibrated classifier).
        filepath: Destination file path (.joblib).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"[calibration] Calibrated pipeline saved -> {filepath}")


def load_calibrated_pipeline(filepath: Path) -> Pipeline:
    """Load a calibrated pipeline from disk.

    Args:
        filepath: Path to the .joblib file.

    Returns:
        Loaded Pipeline.
    """
    return joblib.load(filepath)
