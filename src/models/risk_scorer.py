"""Risk scoring module for fraud detection.

Provides the ``predict_risk`` function which takes raw transaction features,
applies the calibrated pipeline, and returns:
  - fraud_probability  : calibrated sigmoid probability in [0.0, 1.0]
  - review_decision    : bool — True means flag for analyst review
  - risk_tier          : 'Low', 'Medium', or 'High'

Risk tier thresholds (defaults):
  Low    : fraud_probability < 0.10
  Medium : 0.10 <= fraud_probability < 0.50
  High   : fraud_probability >= 0.50

These cutoffs are configurable via ``low_cutoff`` and ``high_cutoff`` args.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


# Required columns for input validation
REQUIRED_COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input_schema(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame has the required columns and dtypes.

    Args:
        df: Input transaction DataFrame.

    Raises:
        ValueError: If required columns are missing or have non-numeric dtype.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"predict_risk: Missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    non_numeric = [
        c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise ValueError(
            f"predict_risk: Non-numeric dtype in columns: {non_numeric}. "
            "All feature columns must be numeric."
        )


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def predict_risk(
    df: pd.DataFrame,
    calibrated_pipeline: Pipeline,
    threshold: float,
    low_cutoff: float = 0.10,
    high_cutoff: float = 0.50,
) -> pd.DataFrame:
    """Score transactions with calibrated fraud probabilities and risk tiers.

    Args:
        df: Input DataFrame containing at least Time, Amount, V1-V28.
            Additional columns are ignored.
        calibrated_pipeline: Fitted Pipeline (preprocessor + calibrated clf).
        threshold: Decision threshold for the binary review flag.
            Transactions with fraud_probability >= threshold are flagged.
        low_cutoff: Probabilities below this value -> risk_tier = 'Low'.
        high_cutoff: Probabilities at or above this value -> risk_tier = 'High'.
            Values in [low_cutoff, high_cutoff) -> risk_tier = 'Medium'.

    Returns:
        DataFrame with the same index as ``df`` and three new columns:
        - fraud_probability (float): Calibrated probability of fraud.
        - review_decision (bool): True = flag for analyst review.
        - risk_tier (str): 'Low', 'Medium', or 'High'.

    Raises:
        ValueError: If required columns are missing or non-numeric.
        ValueError: If threshold, low_cutoff, or high_cutoff are out of range.
    """
    # --- Validate parameters ---
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    if not (0.0 <= low_cutoff < high_cutoff <= 1.0):
        raise ValueError(
            f"low_cutoff ({low_cutoff}) must be < high_cutoff ({high_cutoff}) "
            "and both in [0, 1]."
        )

    # --- Validate input schema ---
    validate_input_schema(df)

    # --- Extract only the required feature columns ---
    X = df[REQUIRED_COLUMNS].copy()

    # --- Predict calibrated probabilities ---
    fraud_proba = calibrated_pipeline.predict_proba(X)[:, 1]

    # --- Bounds check ---
    if not ((fraud_proba >= 0.0).all() and (fraud_proba <= 1.0).all()):
        raise RuntimeError(
            "calibrated_pipeline returned probabilities outside [0, 1]. "
            "Check calibration step."
        )

    # --- Build output DataFrame ---
    result = pd.DataFrame(index=df.index)
    result["fraud_probability"] = fraud_proba.astype(np.float64)
    result["review_decision"] = fraud_proba >= threshold

    # Assign risk tiers
    tiers = np.where(
        fraud_proba >= high_cutoff,
        "High",
        np.where(fraud_proba >= low_cutoff, "Medium", "Low"),
    )
    result["risk_tier"] = tiers

    return result


# ---------------------------------------------------------------------------
# Threshold metadata serialization
# ---------------------------------------------------------------------------

def save_threshold_metadata(
    threshold: float,
    model_name: str,
    objective: str,
    metrics: Dict,
    filepath: Path,
) -> None:
    """Save threshold and scoring metadata to a JSON file.

    Args:
        threshold: Selected operational threshold.
        model_name: Name of the winning model.
        objective: Objective used to select the threshold.
        metrics: Metric dict (precision, recall, f1, fp, fn, etc.).
        filepath: Destination JSON file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "optimal_threshold": threshold,
        "selection_objective": objective,
        "metrics_at_threshold": {
            k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in metrics.items()
        },
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[risk_scorer] Threshold metadata saved -> {filepath}")


def load_threshold_metadata(filepath: Path) -> Dict:
    """Load threshold metadata from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Loaded dict.
    """
    with open(filepath) as f:
        return json.load(f)
