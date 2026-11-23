"""Model comparison, multi-model PR curve visualization, and feature importance analysis.

Benchmarks multiple classifiers on the identical validation split and generates
publication-quality comparison charts using Matplotlib and Plotly.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.models.random_forest import get_rf_feature_importances
from src.models.xgboost_model import get_xgb_feature_importances


def benchmark_models(
    models: Dict[str, Pipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Benchmark multiple model pipelines on the identical validation split.

    Args:
        models: Dictionary of model_name -> fitted Pipeline.
        X_val: Validation feature DataFrame.
        y_val: Validation ground truth target.
        threshold: Classification decision threshold.

    Returns:
        pd.DataFrame containing comparative metrics across all models.
    """
    rows = []
    for name, pipeline in models.items():
        probs = pipeline.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)

        pr_auc = float(average_precision_score(y_val, probs))
        roc_auc = float(roc_auc_score(y_val, probs))
        precision = float(precision_score(y_val, preds, zero_division=0))
        recall = float(recall_score(y_val, preds, zero_division=0))
        f1 = float(f1_score(y_val, preds, zero_division=0))

        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        rows.append({
            "model": name,
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "fraud_detected_ratio": f"{tp}/{int(y_val.sum())}",
            "threshold": threshold,
        })

    df_comp = pd.DataFrame(rows).sort_values(by="pr_auc", ascending=False).reset_index(drop=True)
    return df_comp


def plot_multi_model_pr_curves(
    models: Dict[str, Pipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    output_dir: Union[str, Path] = "reports/figures",
) -> Tuple[str, str]:
    """Plot Precision-Recall curves for all evaluated models on the same figure.

    Args:
        models: Dictionary of model_name -> fitted Pipeline.
        X_val: Validation features.
        y_val: Validation targets.
        output_dir: Directory to save figure.

    Returns:
        Tuple of (png_path, html_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_rate = float(y_val.mean())
    colors = ["#2b5c8f", "#2ca02c", "#d9534f", "#9467bd"]

    # 1. Matplotlib
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for idx, (name, pipeline) in enumerate(models.items()):
        probs = pipeline.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        pr_auc = average_precision_score(y_val, probs)
        color = colors[idx % len(colors)]
        ax.plot(recall, precision, color=color, linewidth=2.5, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    ax.axhline(baseline_rate, color="gray", linestyle="--", linewidth=1.2, label=f"Random Chance ({baseline_rate*100:.3f}%)")
    ax.set_xlabel("Recall (Detection Rate)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Model Benchmarking: Validation Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    mpl_path = out_dir / "11_precision_recall_comparison.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly
    fig_plotly = go.Figure()
    for idx, (name, pipeline) in enumerate(models.items()):
        probs = pipeline.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, probs)
        pr_auc = average_precision_score(y_val, probs)
        fig_plotly.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"{name} (PR-AUC = {pr_auc:.4f})",
            line=dict(width=2.5)
        ))

    fig_plotly.add_hline(y=baseline_rate, line_dash="dash", line_color="gray", annotation_text="Random Chance")
    fig_plotly.update_layout(
        title="Validation Precision-Recall Curves Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
    )
    plotly_path = out_dir / "11_precision_recall_comparison.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_feature_importances_comparison(
    rf_pipeline: Pipeline,
    xgb_pipeline: Pipeline,
    output_dir: Union[str, Path] = "reports/figures",
    top_n: int = 12,
) -> Tuple[str, str]:
    """Generate side-by-side feature importance comparisons for RF and XGBoost.

    Args:
        rf_pipeline: Fitted Random Forest pipeline.
        xgb_pipeline: Fitted XGBoost pipeline.
        output_dir: Directory to save charts.
        top_n: Number of top features to display.

    Returns:
        Tuple of (png_path, html_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf_imp = get_rf_feature_importances(rf_pipeline).head(top_n).sort_values(by="importance", ascending=True)
    xgb_imp = get_xgb_feature_importances(xgb_pipeline).head(top_n).sort_values(by="importance", ascending=True)

    # 1. Matplotlib Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.barh(rf_imp["feature"], rf_imp["importance"], color="#2ca02c", edgecolor="black", alpha=0.85)
    ax1.set_title(f"Random Forest (Top {top_n} Features - Gini Impurity)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    ax2.barh(xgb_imp["feature"], xgb_imp["importance"], color="#d9534f", edgecolor="black", alpha=0.85)
    ax2.set_title(f"XGBoost (Top {top_n} Features - Gain)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Relative Information Gain", fontsize=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Feature Importance Comparison: Tree Ensemble Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mpl_path = out_dir / "12_feature_importance_rf_xgb.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly Interactive Bar Subplots
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Bar(
        x=rf_imp["importance"],
        y=rf_imp["feature"],
        orientation="h",
        name="Random Forest",
        marker_color="#2ca02c",
    ))
    fig_plotly.add_trace(go.Bar(
        x=xgb_imp["importance"],
        y=xgb_imp["feature"],
        orientation="h",
        name="XGBoost",
        marker_color="#d9534f",
    ))
    fig_plotly.update_layout(
        title=f"Top {top_n} Feature Importances: Random Forest vs XGBoost",
        barmode="group",
        template="plotly_white",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
    )
    plotly_path = out_dir / "12_feature_importance_rf_xgb.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def save_comparison_metrics(
    comparison_df: pd.DataFrame,
    json_path: Union[str, Path] = "artifacts/model_comparison_metrics.json",
    csv_path: Union[str, Path] = "artifacts/model_comparison_metrics.csv",
) -> Tuple[Path, Path]:
    """Serialize comparison metrics to both JSON and CSV files."""
    j_path = Path(json_path)
    c_path = Path(csv_path)

    j_path.parent.mkdir(parents=True, exist_ok=True)
    c_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_df.to_json(j_path, orient="records", indent=2)
    comparison_df.to_csv(c_path, index=False)

    return j_path, c_path
