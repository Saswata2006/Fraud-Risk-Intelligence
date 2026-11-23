"""Model evaluation and diagnostic visualization module.

Computes imbalanced classification metrics (PR-AUC, ROC-AUC, Recall,
Precision, F1, Confusion Matrix) and generates diagnostic plots including
Precision-Recall curve, Confusion Matrix heatmap, and probability score distributions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
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
    roc_curve,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics on validation data.

    Args:
        pipeline: Fitted model pipeline.
        X_val: Validation feature DataFrame.
        y_val: Validation ground truth binary labels.
        threshold: Decision threshold for binary classification (default 0.5).

    Returns:
        Dictionary of computed evaluation metrics.
    """
    probabilities = pipeline.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_val, probabilities))
    roc_auc = float(roc_auc_score(y_val, probabilities))
    precision = float(precision_score(y_val, predictions, zero_division=0))
    recall = float(recall_score(y_val, predictions, zero_division=0))
    f1 = float(f1_score(y_val, predictions, zero_division=0))

    cm = confusion_matrix(y_val, predictions)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "threshold": threshold,
        "average_precision_pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "total_validation_samples": len(y_val),
        "actual_fraud_samples": int(y_val.sum()),
        "actual_legit_samples": int((y_val == 0).sum()),
        "predicted_fraud_samples": int(predictions.sum()),
    }
    return metrics


def plot_precision_recall_curve(
    y_val: pd.Series,
    probabilities: np.ndarray,
    output_dir: Union[str, Path] = "reports/figures",
) -> Tuple[str, str]:
    """Generate and save Precision-Recall curve in PNG and HTML formats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)
    baseline_rate = y_val.mean()

    # 1. Matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="#2b5c8f", linewidth=2.5, label=f"Logistic Regression (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline_rate, color="#d9534f", linestyle="--", linewidth=1.5, label=f"Random Baseline ({baseline_rate*100:.3f}%)")
    ax.set_xlabel("Recall (Detection Rate)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Validation Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    mpl_path = out_dir / "08_precision_recall_curve.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly
    fig_plotly = px.line(
        x=recall,
        y=precision,
        labels={"x": "Recall", "y": "Precision"},
        title=f"Validation Precision-Recall Curve (PR-AUC = {pr_auc:.4f})"
    )
    fig_plotly.add_hline(y=baseline_rate, line_dash="dash", line_color="red", annotation_text=f"Baseline ({baseline_rate*100:.3f}%)")
    fig_plotly.update_layout(template="plotly_white")
    plotly_path = out_dir / "08_precision_recall_curve.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_confusion_matrix_heatmap(
    cm_dict: Dict[str, int],
    threshold: float = 0.5,
    output_dir: Union[str, Path] = "reports/figures",
) -> Tuple[str, str]:
    """Generate and save Confusion Matrix heatmap in PNG and HTML formats."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    cm_matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN\n" + f"{tn:,}", "FP\n" + f"{fp:,}"], ["FN\n" + f"{fn:,}", "TP\n" + f"{tp:,}"]]

    # 1. Matplotlib
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm_matrix, cmap="Blues", alpha=0.8)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center", color="black" if cm_matrix[i, j] < (tn / 2) else "white", fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"], fontsize=11)
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"], fontsize=11)
    ax.set_title(f"Confusion Matrix (Validation @ Threshold = {threshold})", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    mpl_path = out_dir / "09_confusion_matrix.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly Heatmap
    fig_plotly = go.Figure(data=go.Heatmap(
        z=cm_matrix,
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        text=[[f"TN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]],
        texttemplate="%{text}",
        colorscale="Blues",
    ))
    fig_plotly.update_layout(
        title=f"Confusion Matrix Heatmap (Threshold = {threshold})",
        template="plotly_white"
    )
    plotly_path = out_dir / "09_confusion_matrix.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def plot_prediction_score_distribution(
    y_val: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    output_dir: Union[str, Path] = "reports/figures",
) -> Tuple[str, str]:
    """Generate and save predicted fraud probability score distributions by actual class."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_legit = probabilities[y_val == 0]
    prob_fraud = probabilities[y_val == 1]

    # 1. Matplotlib Log-Density Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale with threshold line
    ax1.hist(prob_legit, bins=50, alpha=0.6, label="Actual Legit", color="#2b5c8f", density=True)
    ax1.hist(prob_fraud, bins=50, alpha=0.7, label="Actual Fraud", color="#d9534f", density=True)
    ax1.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Default Threshold ({threshold})")
    ax1.set_xlabel("Predicted Fraud Probability", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Probability Distribution (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper center")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Log scale Y
    ax2.hist(prob_legit, bins=50, alpha=0.6, label="Actual Legit", color="#2b5c8f", log=True)
    ax2.hist(prob_fraud, bins=50, alpha=0.7, label="Actual Fraud", color="#d9534f", log=True)
    ax2.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")
    ax2.set_xlabel("Predicted Fraud Probability", fontsize=11)
    ax2.set_ylabel("Count (Log Scale)", fontsize=11)
    ax2.set_title("Prediction Scores by Class (Log-Count Scale)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Validation Prediction Score Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mpl_path = out_dir / "10_prediction_score_distribution.png"
    plt.savefig(mpl_path, dpi=300)
    plt.close()

    # 2. Plotly Boxplot / Histogram
    score_df = pd.DataFrame({
        "Probability": probabilities,
        "Class": y_val.map({0: "Legitimate", 1: "Fraudulent"})
    })
    fig_plotly = px.box(
        score_df,
        x="Class",
        y="Probability",
        color="Class",
        color_discrete_map={"Legitimate": "#2b5c8f", "Fraudulent": "#d9534f"},
        title="Predicted Fraud Probability by True Class"
    )
    fig_plotly.update_layout(template="plotly_white")
    plotly_path = out_dir / "10_prediction_score_distribution.html"
    fig_plotly.write_html(plotly_path)

    return str(mpl_path), str(plotly_path)


def save_metrics_to_json(
    metrics: Dict[str, Any],
    filepath: Union[str, Path] = "artifacts/baseline_metrics.json",
) -> Path:
    """Save computed validation metrics to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path
