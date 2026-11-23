"""Decision threshold optimization for fraud detection models.

Evaluates precision, recall, F1, false positives, false negatives, and an
illustrative financial cost across a range of classification thresholds using
*validation data only*.  Selects an operational threshold based on a stated
objective so the choice is transparent and reproducible.

COST ASSUMPTIONS (illustrative — clearly stated):
  - FP cost (false positive = block a legitimate transaction): $10
    (manual review cost + customer friction).
  - FN cost (false negative = miss a fraud): $500
    (average fraud loss per transaction, illustrative only).
These numbers are NOT derived from real business data; they are placeholders
to demonstrate cost-sensitive threshold selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline


FP_COST: float = 10.0    # USD per false positive (illustrative)
FN_COST: float = 500.0   # USD per false negative (illustrative)

DEFAULT_THRESHOLDS: np.ndarray = np.concatenate([
    np.arange(0.005, 0.05, 0.005),   # Sub-0.05 range (audit extended search)
    np.arange(0.05, 0.96, 0.01),     # Standard range with fine 0.01 step
])


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def compute_threshold_metrics(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute classification metrics over a grid of decision thresholds.

    Args:
        pipeline: Fitted sklearn Pipeline with ``predict_proba`` method.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series (0/1).
        thresholds: Array of threshold values in (0, 1). Defaults to
            ``DEFAULT_THRESHOLDS`` spanning [0.005, 0.95].

    Returns:
        DataFrame with columns: threshold, precision, recall, f1,
        tp, fp, tn, fn, financial_cost.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    proba = pipeline.predict_proba(X_val)[:, 1]
    y_true = y_val.values

    rows: List[Dict] = []
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        cost = compute_financial_cost(
            {"fp": fp, "fn": fn},
            fp_cost=FP_COST,
            fn_cost=FN_COST,
        )

        rows.append(
            {
                "threshold": float(thr),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "financial_cost": float(cost),
            }
        )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    metrics_df: pd.DataFrame,
    objective: str = "f1",
) -> Tuple[float, pd.Series]:
    """Return the threshold that optimises the stated objective.

    Args:
        metrics_df: DataFrame produced by ``compute_threshold_metrics``.
        objective: Column name to maximise.  One of:
            - 'f1' (default): maximise F1-score.
            - 'recall': maximise recall.
            - 'precision': maximise precision.
            - 'financial_cost': minimise total illustrative cost.

    Returns:
        Tuple of (optimal_threshold, metric_row_at_that_threshold).
    """
    if objective == "financial_cost":
        idx = metrics_df["financial_cost"].idxmin()
    else:
        idx = metrics_df[objective].idxmax()

    best_row = metrics_df.loc[idx]
    optimal_threshold = float(best_row["threshold"])
    print(
        f"[threshold] Optimal threshold ({objective}) = {optimal_threshold:.3f} | "
        f"P={best_row['precision']:.4f} R={best_row['recall']:.4f} "
        f"F1={best_row['f1']:.4f} FP={int(best_row['fp'])} FN={int(best_row['fn'])}"
    )
    return optimal_threshold, best_row


def compare_f1_and_cost_thresholds(
    metrics_df: pd.DataFrame,
) -> Tuple[Tuple[float, pd.Series], Tuple[float, pd.Series]]:
    """Compare F1-optimal and cost-optimal operating points.

    Args:
        metrics_df: DataFrame produced by ``compute_threshold_metrics``.

    Returns:
        Tuple of ((f1_thr, f1_row), (cost_thr, cost_row)).
    """
    f1_thr, f1_row = find_optimal_threshold(metrics_df, objective="f1")
    cost_thr, cost_row = find_optimal_threshold(metrics_df, objective="financial_cost")
    return (f1_thr, f1_row), (cost_thr, cost_row)


# ---------------------------------------------------------------------------
# Financial cost
# ---------------------------------------------------------------------------

def compute_financial_cost(
    metrics_dict: Dict[str, int],
    fp_cost: float = FP_COST,
    fn_cost: float = FN_COST,
) -> float:
    """Calculate illustrative total financial cost from confusion matrix counts.

    ASSUMPTIONS (illustrative only — not based on real business data):
      - Each false positive costs $10 (manual analyst review + customer friction).
      - Each false negative costs $500 (assumed average fraudulent transaction loss).

    Args:
        metrics_dict: Dict with at least 'fp' and 'fn' integer counts.
        fp_cost: Cost per false positive in USD.
        fn_cost: Cost per false negative in USD.

    Returns:
        Total illustrative financial cost in USD.
    """
    return metrics_dict["fp"] * fp_cost + metrics_dict["fn"] * fn_cost


# ---------------------------------------------------------------------------
# Precision / Recall at k
# ---------------------------------------------------------------------------

def precision_recall_at_k(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    k: int,
) -> Tuple[float, float]:
    """Evaluate precision@k and recall@k for a fixed analyst review budget.

    Sorts transactions by descending fraud probability; the analyst reviews
    only the top-k flagged transactions.

    Args:
        pipeline: Fitted Pipeline with ``predict_proba``.
        X_val: Validation feature DataFrame.
        y_val: Validation target Series.
        k: Review budget (number of transactions to flag).

    Returns:
        Tuple of (precision_at_k, recall_at_k).
    """
    proba = pipeline.predict_proba(X_val)[:, 1]
    top_k_idx = np.argsort(proba)[::-1][:k]
    y_true = y_val.values
    tp = int(y_true[top_k_idx].sum())
    prec_k = tp / k if k > 0 else 0.0
    total_fraud = int(y_true.sum())
    rec_k = tp / total_fraud if total_fraud > 0 else 0.0
    print(f"[threshold] Precision@{k}={prec_k:.4f}  Recall@{k}={rec_k:.4f}")
    return prec_k, rec_k


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_curves(
    metrics_dfs: Dict[str, pd.DataFrame],
    output_dir: Path,
    optimal_thresholds: Optional[Dict[str, float]] = None,
) -> None:
    """Plot precision, recall, and F1 vs threshold for multiple models.

    Args:
        metrics_dfs: Dict mapping model label to metrics DataFrame.
        output_dir: Directory to save figures.
        optimal_thresholds: Optional dict of model label -> optimal threshold
            to annotate on the plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["steelblue", "darkorange", "green"]
    metrics_to_plot = ["precision", "recall", "f1"]
    titles = ["Precision vs Threshold", "Recall vs Threshold", "F1 vs Threshold"]
    y_labels = ["Precision", "Recall", "F1-Score"]

    for col_idx, (metric, title, ylabel) in enumerate(
        zip(metrics_to_plot, titles, y_labels)
    ):
        ax = axes[col_idx]
        for (label, df), color in zip(metrics_dfs.items(), colors):
            ax.plot(df["threshold"], df[metric], label=label, color=color, lw=2)
            if optimal_thresholds and label in optimal_thresholds:
                opt_thr = optimal_thresholds[label]
                opt_val = df.loc[
                    (df["threshold"] - opt_thr).abs().idxmin(), metric
                ]
                ax.axvline(opt_thr, color=color, linestyle="--", alpha=0.6)
                ax.scatter([opt_thr], [opt_val], color=color, zorder=5, s=60)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Threshold")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Threshold Optimization Curves (Validation Set)\n"
        "Dashed verticals = F1-optimal thresholds",
        fontsize=12,
    )
    plt.tight_layout()

    png_path = output_dir / "14_threshold_curves.png"
    html_path = output_dir / "14_threshold_curves.html"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    html_content = f"""<!DOCTYPE html><html><body>
<h2>Threshold Optimization Curves</h2>
<img src="{png_path.name}" style="max-width:1200px;">
</body></html>"""
    html_path.write_text(html_content)
    plt.close(fig)
    print(f"[threshold] Threshold curves saved -> {png_path}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_threshold_metadata(
    threshold: float,
    model_name: str,
    objective: str,
    metrics_row: pd.Series,
    fp_cost_assumption: float,
    fn_cost_assumption: float,
    filepath: Path,
    cost_threshold: Optional[float] = None,
    cost_metrics_row: Optional[pd.Series] = None,
) -> None:
    """Save threshold selection metadata and all metrics to a JSON file.

    Args:
        threshold: Selected operational threshold (e.g. F1-optimal).
        model_name: Name of the winning model.
        objective: Primary objective used to select the threshold.
        metrics_row: Metric row at the optimal threshold.
        fp_cost_assumption: Illustrative FP cost (USD).
        fn_cost_assumption: Illustrative FN cost (USD).
        filepath: Destination JSON file path.
        cost_threshold: Optional cost-optimal threshold.
        cost_metrics_row: Optional metric row at the cost-optimal threshold.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "optimal_threshold": threshold,
        "selection_objective": objective,
        "cost_assumptions": {
            "fp_cost_usd": fp_cost_assumption,
            "fn_cost_usd": fn_cost_assumption,
            "note": (
                "Illustrative costs only — not derived from real business data. "
                "FP = manual review cost; FN = assumed average fraud loss."
            ),
        },
        "metrics_at_threshold": {
            k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
            for k, v in metrics_row.items()
        },
    }

    if cost_threshold is not None and cost_metrics_row is not None:
        payload["cost_optimal_operating_point"] = {
            "optimal_threshold": cost_threshold,
            "metrics": {
                k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
                for k, v in cost_metrics_row.items()
            },
        }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[threshold] Threshold metadata saved -> {filepath}")


def load_threshold_metadata(filepath: Path) -> Dict:
    """Load threshold metadata from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Loaded dict.
    """
    with open(filepath) as f:
        return json.load(f)
