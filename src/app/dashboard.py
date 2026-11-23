"""Four-page analyst dashboard for saved Fraud Risk Intelligence artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Cloud executes this nested file with src/app on sys.path.
# Add the repository root so the src package resolves in every environment.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.charts import calibration_summary_figure, precision_recall_summary_figure, shap_contribution_figure
from src.app.data import ROOT, load_demo_summary, load_demo_transactions, parse_batch_csv, review_threshold, score_batch
from src.explainability.shap_explainer import compute_shap_values
from src.models.calibration import load_calibrated_pipeline


@st.cache_resource(show_spinner="Loading calibrated Random Forest…")
def calibrated_model() -> object:
    """Load the unchanged saved RF pipeline once per Streamlit process."""
    return load_calibrated_pipeline(ROOT / "artifacts/calibrated_rf_pipeline.joblib")


@st.cache_data(show_spinner="Reading demo transactions…")
def demo_transactions() -> pd.DataFrame:
    """Cache the 100-row validation sample shipped with the app."""
    return load_demo_transactions()


@st.cache_data(show_spinner="Reading validation summaries…")
def demo_summary() -> dict:
    """Cache precomputed aggregates and curves; no raw CSV is needed."""
    return load_demo_summary()


def data_intelligence() -> None:
    """Render precomputed descriptive statistics from train and validation."""
    st.title("Data Intelligence")
    st.caption("Exploratory view · chronological train + validation only · final test rows excluded")
    stats = demo_summary()["eda"]
    count = stats["transactions"]
    fraud = stats["fraud"]
    a, b, c, d = st.columns(4)
    a.metric("Transactions", f"{count:,}")
    b.metric("Fraud cases", f"{fraud:,}")
    c.metric("Fraud prevalence", f"{fraud / count:.3%}")
    d.metric("Median amount", f"{stats['median_amount']:,.2f}")
    st.plotly_chart(px.bar(pd.DataFrame(stats["class_counts"]), x="partition", y="Transactions", color="Class", barmode="group", log_y=True, title="Class distribution by partition", template="plotly_white"), width="stretch")
    left, right = st.columns(2)
    with left:
        hourly = pd.DataFrame(stats["hourly"])
        st.plotly_chart(px.line(hourly, x="elapsed_hour", y=["transactions", "fraud"], title="Transactions and fraud by elapsed hour", labels={"value": "Count", "elapsed_hour": "Hours since first transaction"}, template="plotly_white"), width="stretch")
    with right:
        amount = pd.DataFrame(stats["amount_bands"])
        st.plotly_chart(px.bar(amount, x="Amount", y="fraud_rate", title="Fraud rate by amount band", labels={"fraud_rate": "Fraud rate", "Amount": "Amount band"}, template="plotly_white"), width="stretch")
    st.caption("Time is elapsed seconds; V1–V28 are anonymized components. Amount units are those supplied by the dataset.")


def model_performance() -> None:
    """Render saved validation metrics and live validation diagnostics."""
    st.title("Model Performance")
    st.caption("Validation-only results · 0.50 threshold for baseline model comparison · final test set untouched")
    summary = demo_summary()
    comparison = pd.DataFrame(summary["model_comparison"])
    shown = comparison[["model", "pr_auc", "precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives"]].rename(columns={"model": "Model", "pr_auc": "Average precision (PR-AUC)", "f1_score": "F1", "true_positives": "TP", "false_positives": "FP", "false_negatives": "FN"})
    st.dataframe(shown, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(comparison, x="model", y="pr_auc", title="Validation average precision", labels={"model": "Model", "pr_auc": "Average precision"}, template="plotly_white"), width="stretch")
    ap = dict(zip(comparison["model"].str.replace(" (Baseline)", "", regex=False), comparison["pr_auc"]))
    st.plotly_chart(precision_recall_summary_figure(summary["validation_pr_curves"], ap, summary["validation_fraud_prevalence"]), width="stretch")
    st.subheader("Probability calibration · validation only")
    calibration = summary["calibration_metrics"]
    cal_table = pd.DataFrame([{"Model": "Random Forest", **calibration["rf"]}, {"Model": "XGBoost", **calibration["xgb"]}])
    st.dataframe(cal_table.rename(columns={"brier_raw": "Brier · raw", "brier_cal": "Brier · calibrated", "logloss_raw": "Log loss · raw", "logloss_cal": "Log loss · calibrated"}), hide_index=True, width="stretch")
    st.plotly_chart(calibration_summary_figure(summary["validation_calibration_curves"]), width="stretch")
    metadata = summary["threshold_metadata"]
    metrics = metadata["metrics_at_threshold"]
    st.subheader("Selected review operating point · validation only")
    a, b, c, d = st.columns(4)
    a.metric("Review threshold", f"{metadata['optimal_threshold']:.3f}")
    b.metric("Precision", f"{metrics['precision']:.1%}")
    c.metric("Recall", f"{metrics['recall']:.1%}")
    d.metric("F1", f"{metrics['f1']:.3f}")
    st.caption(f"TP {int(metrics['tp'])} · FP {int(metrics['fp'])} · FN {int(metrics['fn'])}. Threshold selected on validation via {metadata['selection_objective'].upper()}; these figures are validation-only and may be optimistic. Financial cost assumptions in artifacts are illustrative.")


def fraud_prediction() -> None:
    """Score a selected validation sample and explain the raw RF output."""
    st.title("Fraud Prediction")
    st.caption("Demonstration samples come from validation only. The calibrated RF pipeline scores raw features; no model fitting occurs.")
    samples = demo_transactions()
    sample_type = st.radio("Sample pool", ["Any validation transaction", "Validation fraud examples", "Validation legitimate examples"], horizontal=True)
    if sample_type == "Validation fraud examples":
        pool = samples.iloc[demo_summary()["sample"]["fraud_example_positions"]]
    elif sample_type == "Validation legitimate examples":
        pool = samples.iloc[demo_summary()["sample"]["legitimate_example_positions"]]
    else:
        pool = samples
    position = st.number_input("Sample position in selected pool", min_value=0, max_value=len(pool) - 1, value=0, step=1)
    selected = pool.iloc[[int(position)]]
    features = selected
    calibrated = calibrated_model()
    threshold = review_threshold()
    scored = score_batch(features, calibrated, threshold).iloc[0]
    a, b, c = st.columns(3)
    a.metric("Calibrated fraud probability", f"{scored['fraud_probability']:.2%}")
    b.metric("Review decision", "Flag for review" if scored["review_decision"] else "No review flag")
    c.metric("Risk tier", scored["risk_tier"])
    st.caption(f"Review flag: calibrated probability ≥ {threshold:.3f} (validation selected). Risk tiers: Low <10%; Medium 10–<50%; High ≥50%. The review flag is independent of the tier label.")
    st.dataframe(features, hide_index=True, width="stretch")
    if st.button("Explain this transaction", type="primary"):
        with st.spinner("Computing TreeSHAP contributions…"):
            preprocessor = calibrated.named_steps["preprocessor"]
            raw_rf = calibrated.named_steps["classifier"].calibrated_classifiers_[0].estimator.estimator
            values, _, names = compute_shap_values(preprocessor, raw_rf, features, model_type="random_forest")
            st.plotly_chart(shap_contribution_figure(values[0], names), width="stretch")
            st.caption("SHAP explains the underlying RF fraud output before sigmoid calibration. Its contributions do not add up to the displayed calibrated probability.")


def batch_screening() -> None:
    """Validate a user CSV and offer calibrated RF screening results."""
    st.title("Batch Screening")
    st.caption("Upload transactions with Time, Amount, and V1–V28 numeric columns. Optional identifier columns are preserved. No labels are needed.")
    st.download_button("Download 100-row validation demo CSV", demo_transactions().to_csv(index=False).encode("utf-8"), "validation_transactions_100.csv", "text/csv")
    st.info("Download the demo CSV above, then upload that same file below to try Batch Screening. It contains 100 validation transactions and the 30 required numerical features, with no labels. Its 10 fraud and 90 legitimate examples are intentionally sampled and do not represent fraud prevalence.")
    upload = st.file_uploader("Transaction CSV", type="csv")
    if upload is None:
        return
    try:
        frame = parse_batch_csv(upload.getvalue())
        result = score_batch(frame, calibrated_model(), review_threshold())
    except (ValueError, pd.errors.ParserError, UnicodeError) as error:
        st.error(f"Could not score CSV: {error}")
        return
    st.success(f"Scored {len(result):,} transactions with the saved calibrated Random Forest.")
    a, b, c = st.columns(3)
    a.metric("Review flags", f"{int(result['review_decision'].sum()):,}")
    b.metric("High risk tier", f"{int((result['risk_tier'] == 'High').sum()):,}")
    c.metric("Review threshold", f"{review_threshold():.3f}")
    st.caption(f"Review flag: calibrated probability ≥ {review_threshold():.3f} (validation selected). Risk tiers: Low <10%; Medium 10–<50%; High ≥50%. The review flag is independent of the tier label.")
    st.dataframe(result, hide_index=True, width="stretch")
    st.download_button("Download scored CSV", result.to_csv(index=False).encode("utf-8"), "fraud_screening_results.csv", "text/csv")


def main() -> None:
    """Set page configuration and dispatch the four dashboard pages."""
    st.set_page_config(page_title="Fraud Risk Intelligence", page_icon="🛡️", layout="wide")
    st.sidebar.title("Fraud Risk Intelligence")
    st.sidebar.caption("Analyst workspace · saved Day 1–4 artifacts")
    pages = {"Data Intelligence": data_intelligence, "Model Performance": model_performance, "Fraud Prediction": fraud_prediction, "Batch Screening": batch_screening}
    selection = st.sidebar.radio("Navigate", list(pages))
    st.sidebar.divider()
    st.sidebar.caption("Validation metrics only. This app uses compact demo data and never loads the final test partition.")
    try:
        pages[selection]()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Dashboard data or artifact unavailable: {error}")


if __name__ == "__main__":
    main()
