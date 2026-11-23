r"""SHAP explainability module for tree-based fraud detection models.

Implements global (summary + feature importance) and local (waterfall)
SHAP explanations using ``shap.TreeExplainer``.

IMPORTANT NOTE ON WHAT SHAP EXPLAINS:
  SHAP values computed here explain the **raw tree model's decision margin (log-odds output)**
  (XGBClassifier margin scores / RF raw leaf values), NOT the
  sigmoid-calibrated probabilities produced by CalibratedClassifierCV.
  
  Platt calibration applies a monotonic sigmoid mapping. While monotonicity preserves
  the global ordinal ranking of overall transaction risk, feature additivity
  (local efficiency $\sum \phi_i = f(x) - E[f(x)]$) holds strictly on the raw margin
  scale. It does NOT translate into additive probability contributions because the
  sigmoid transformation is non-linear and saturates in the tails. Consequently,
  SHAP values reflect the raw tree ensemble's feature contributions before calibration.

FEATURE ALIGNMENT:
  The preprocessor produces 30 output features in order:
    [Time, Amount, V1, V2, ..., V28]
  This order is retrieved via ``preprocessor.get_feature_names_out()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    fitted_preprocessor: Any,
    raw_clf: Any,
    X_sample: pd.DataFrame,
    model_type: str = "xgboost",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute SHAP values using TreeExplainer on the raw (uncalibrated) model.

    SHAP explains the RAW MODEL OUTPUT (log-odds / raw leaf scores), not the
    calibrated probabilities.  This is intentional: TreeExplainer requires
    direct access to the tree structure, which is not available through
    CalibratedClassifierCV.

    Args:
        fitted_preprocessor: Fitted ColumnTransformer (from calibration module).
        raw_clf: Raw fitted classifier (RandomForestClassifier or XGBClassifier).
        X_sample: DataFrame of transactions to explain (use a representative
            subsample for speed; e.g., 2000 rows from validation set).
        model_type: 'xgboost' or 'random_forest'.  Used for expected value
            extraction and display labelling.

    Returns:
        Tuple of:
        - shap_values: np.ndarray of shape (n_samples, n_features).
          For binary classifiers this is the SHAP values for class=1 (fraud).
        - X_trans: np.ndarray of transformed features (same shape).
        - feature_names: List of feature name strings aligned with columns.
    """
    # Transform features (preprocessor is already fitted)
    X_trans = fitted_preprocessor.transform(X_sample)

    # Retrieve feature names from the preprocessor
    try:
        feature_names = list(fitted_preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback if get_feature_names_out is unavailable
        feature_names = [f"feature_{i}" for i in range(X_trans.shape[1])]

    # Build TreeExplainer
    explainer = shap.TreeExplainer(raw_clf)
    shap_output = explainer.shap_values(X_trans)

    # For RF with sklearn API, shap_values can return a list [class0, class1] or 3D array (N, M, 2)
    # For XGBoost it returns a 2-D array for the positive class margin
    if isinstance(shap_output, list):
        shap_values = shap_output[1]
    elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
        shap_values = shap_output[:, :, 1]
    else:
        shap_values = shap_output

    print(
        f"[shap] SHAP values computed | shape={shap_values.shape} | "
        f"features={len(feature_names)} | model_type={model_type}"
    )
    print(
        "[shap] NOTE: These SHAP values explain raw tree log-odds, "
        "NOT calibrated probabilities."
    )
    return shap_values, X_trans, feature_names


# ---------------------------------------------------------------------------
# Global summary plot
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values: np.ndarray,
    X_trans: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    max_display: int = 20,
    model_label: str = "Model",
) -> None:
    """Generate a global SHAP beeswarm summary plot.

    Shows the distribution of SHAP values for each feature across all
    explained samples.  Positive SHAP = pushes toward FRAUD; Negative = LEGIT.

    SHAP values explain raw model log-odds (pre-calibration).

    Args:
        shap_values: Array of shape (n_samples, n_features).
        X_trans: Transformed feature matrix (n_samples, n_features).
        feature_names: Feature name list aligned with columns.
        output_dir: Directory to save figures.
        max_display: Maximum number of features to display.
        model_label: Model name string for the plot title.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_trans,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(
        f"SHAP Summary Plot — {model_label}\n"
        "(explains raw model log-odds, NOT calibrated probabilities)",
        fontsize=11,
        pad=12,
    )
    plt.tight_layout()

    png_path = output_dir / "15_shap_summary.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[shap] SHAP summary plot saved -> {png_path}")


# ---------------------------------------------------------------------------
# Global feature importance (mean |SHAP|)
# ---------------------------------------------------------------------------

def plot_shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    top_n: int = 20,
    model_label: str = "Model",
) -> pd.DataFrame:
    """Generate a bar chart of global feature importance (mean |SHAP| values).

    SHAP values explain raw model log-odds (pre-calibration).

    Args:
        shap_values: Array of shape (n_samples, n_features).
        feature_names: Feature name list aligned with columns.
        output_dir: Directory to save figures.
        top_n: Number of top features to display.
        model_label: Model name for the plot title.

    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["mean_abs_shap"][::-1],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP Value| (impact on raw log-odds)")
    ax.set_title(
        f"SHAP Feature Importance — {model_label}\n"
        f"(Top {top_n} features; explains raw model log-odds, NOT calibrated probabilities)",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    png_path = output_dir / "16_shap_feature_importance.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] SHAP feature importance saved -> {png_path}")
    return importance_df


# ---------------------------------------------------------------------------
# Local waterfall plots
# ---------------------------------------------------------------------------

def plot_shap_waterfall_local(
    shap_values: np.ndarray,
    X_trans: np.ndarray,
    feature_names: List[str],
    expected_value: float,
    sample_idx: int,
    output_dir: Path,
    label: str = "transaction",
    true_label: Optional[int] = None,
    pred_proba: Optional[float] = None,
) -> None:
    """Generate a local SHAP waterfall plot for a single transaction.

    Shows how each feature pushes the model output above or below the
    baseline (expected) value for one specific transaction.

    SHAP values explain raw model log-odds (pre-calibration).

    Args:
        shap_values: Array of shape (n_samples, n_features).
        X_trans: Transformed feature matrix.
        feature_names: Feature name list.
        expected_value: SHAP expected value (base value = average model output).
        sample_idx: Index of the transaction to explain.
        output_dir: Directory to save figures.
        label: Short label for the filename (e.g., 'fraud', 'legit').
        true_label: Ground-truth class label (0 or 1) for annotation.
        pred_proba: Calibrated probability (for annotation only).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sv = shap_values[sample_idx]
    xv = X_trans[sample_idx]

    explanation = shap.Explanation(
        values=sv,
        base_values=expected_value,
        data=xv,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(explanation, max_display=15, show=False)

    title_lines = [
        f"SHAP Waterfall — {label.capitalize()} Transaction (index {sample_idx})",
        "(explains raw model log-odds, NOT calibrated probability)",
    ]
    if true_label is not None:
        title_lines.append(f"True Label: {'FRAUD' if true_label == 1 else 'LEGITIMATE'}")
    if pred_proba is not None:
        title_lines.append(f"Calibrated Fraud Probability: {pred_proba:.4f}")

    plt.title("\n".join(title_lines), fontsize=10, pad=10)
    plt.tight_layout()

    filename = f"{'17' if label == 'fraud' else '18'}_shap_waterfall_{label}.png"
    png_path = output_dir / filename
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[shap] Waterfall plot ({label}) saved -> {png_path}")


# ---------------------------------------------------------------------------
# Helper: get expected value from raw classifier
# ---------------------------------------------------------------------------

def get_expected_value(raw_clf: Any, model_type: str = "xgboost") -> float:
    """Extract the SHAP expected value (base rate) from the raw classifier.

    Args:
        raw_clf: Fitted classifier (RF or XGB).
        model_type: 'xgboost' or 'random_forest'.

    Returns:
        Float expected value (average model output on training data).
    """
    explainer = shap.TreeExplainer(raw_clf)
    ev = explainer.expected_value
    # XGBoost returns a scalar; RF returns a list [ev_class0, ev_class1]
    if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
        return float(ev[1])  # RF class 1
    elif isinstance(ev, (list, np.ndarray)):
        return float(ev[0])  # single-element array
    return float(ev)  # XGBoost scalar
