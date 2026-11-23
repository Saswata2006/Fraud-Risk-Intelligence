"""Plotly figures for validation diagnostics and local explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve


def precision_recall_figure(labels: pd.Series, probabilities: dict[str, np.ndarray]) -> go.Figure:
    """Build comparable validation PR curves from saved model predictions."""
    figure = go.Figure()
    for name, values in probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        ap = average_precision_score(labels, values)
        figure.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} · AP {ap:.4f}"))
    figure.add_hline(y=float(labels.mean()), line_dash="dash", line_color="#718096", annotation_text="Fraud prevalence")
    figure.update_layout(title="Validation precision–recall curves", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white", yaxis_range=[0, 1.02], xaxis_range=[0, 1.02])
    return figure


def calibration_figure(labels: pd.Series, probabilities: dict[str, np.ndarray]) -> go.Figure:
    """Build validation reliability curves for raw and calibrated RF outputs."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="Perfect calibration"))
    for name, values in probabilities.items():
        observed, predicted = calibration_curve(labels, values, n_bins=10, strategy="quantile")
        figure.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers", name=name))
    figure.update_layout(title="Validation reliability · quantile bins", xaxis_title="Mean predicted probability", yaxis_title="Observed fraud rate", template="plotly_white")
    return figure


def precision_recall_summary_figure(curves: dict, average_precision: dict[str, float], prevalence: float) -> go.Figure:
    """Plot downsampled validation curves with full-validation AP labels."""
    figure = go.Figure()
    for name, points in curves.items():
        figure.add_trace(go.Scatter(x=points["x"], y=points["y"], mode="lines", name=f"{name} · AP {average_precision[name]:.4f}"))
    figure.add_hline(y=prevalence, line_dash="dash", line_color="#718096", annotation_text="Validation fraud prevalence")
    figure.update_layout(title="Validation PR curves · display points downsampled", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white", yaxis_range=[0, 1.02], xaxis_range=[0, 1.02])
    return figure


def calibration_summary_figure(curves: dict) -> go.Figure:
    """Plot precomputed validation reliability bins without source transactions."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="Perfect calibration"))
    for name, points in curves.items():
        figure.add_trace(go.Scatter(x=points["x"], y=points["y"], mode="lines+markers", name=name))
    figure.update_layout(title="Validation reliability · quantile bins", xaxis_title="Mean predicted probability", yaxis_title="Observed fraud rate", template="plotly_white")
    return figure


def shap_contribution_figure(values: np.ndarray, names: list[str], top_n: int = 12) -> go.Figure:
    """Show the largest local RF TreeSHAP contributions to raw fraud output."""
    indices = np.argsort(np.abs(values))[-top_n:]
    data = pd.DataFrame({"feature": np.asarray(names)[indices], "contribution": values[indices]})
    data = data.sort_values("contribution")
    figure = px.bar(data, x="contribution", y="feature", orientation="h", color="contribution", color_continuous_scale="RdBu_r")
    figure.update_layout(title="Local TreeSHAP contributions · raw RF fraud output", xaxis_title="Contribution to raw RF fraud score", yaxis_title="", template="plotly_white", coloraxis_showscale=False)
    return figure
