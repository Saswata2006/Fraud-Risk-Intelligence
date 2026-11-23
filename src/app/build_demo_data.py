"""Build small, reproducible public demo inputs from train/validation only.

Run explicitly in a development checkout containing the ignored raw CSV and
saved Day 1–4 models. This script never reads final test rows or fits models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src.app.data import ROOT, TRAIN_ROWS, load_json_artifact, load_visible_data
from src.models.risk_scorer import REQUIRED_COLUMNS
from src.models.baseline import load_model as load_baseline_model
from src.models.calibration import load_calibrated_pipeline
from src.models.random_forest import load_rf_model
from src.models.xgboost_model import load_xgb_model

DEMO_DIR = ROOT / "data/demo"


def compact_curve(x: np.ndarray, y: np.ndarray, max_points: int = 600) -> dict[str, list[float]]:
    """Reduce display points without altering the saved validation metrics."""
    positions = np.unique(np.linspace(0, len(x) - 1, min(len(x), max_points), dtype=int))
    return {"x": x[positions].tolist(), "y": y[positions].tolist()}


def build_demo_data(output_dir: Path = DEMO_DIR) -> tuple[Path, Path]:
    """Write a 100-row feature CSV and aggregate validation dashboard JSON."""
    visible = load_visible_data()
    validation = visible.iloc[TRAIN_ROWS:]
    fraud = validation.loc[validation["Class"] == 1].sample(n=10, random_state=42)
    legitimate = validation.loc[validation["Class"] == 0].sample(n=90, random_state=42)
    selected = pd.concat([fraud, legitimate]).sort_index()
    sample = selected[REQUIRED_COLUMNS].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "validation_transactions_100.csv"
    sample.to_csv(csv_path, index=False)

    class_counts = visible.groupby(["partition", "Class"]).size().reset_index(name="Transactions")
    class_counts["Class"] = class_counts["Class"].map({0: "Legitimate", 1: "Fraud"})
    hourly = visible.assign(elapsed_hour=(visible["Time"] // 3600).astype(int)).groupby("elapsed_hour").agg(transactions=("Class", "size"), fraud=("Class", "sum")).reset_index()
    bands = pd.cut(visible["Amount"], bins=[-0.01, 10, 50, 100, 500, float("inf")], labels=["0–10", "10–50", "50–100", "100–500", "500+"])
    amount = visible.groupby(bands, observed=True)["Class"].agg(["sum", "count"]).reset_index()
    amount["fraud_rate"] = amount["sum"] / amount["count"]

    features = validation[REQUIRED_COLUMNS]
    labels = validation["Class"]
    model_paths = {
        "Logistic Regression": (load_baseline_model, "baseline_logistic_regression.joblib"),
        "Random Forest": (load_rf_model, "random_forest_pipeline.joblib"),
        "XGBoost": (load_xgb_model, "xgboost_pipeline.joblib"),
    }
    curves = {}
    for name, (loader, filename) in model_paths.items():
        model = loader(ROOT / "artifacts" / filename)
        probabilities = model.predict_proba(features)[:, 1]
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        curves[name] = compact_curve(recall, precision)
    calibrated = load_calibrated_pipeline(ROOT / "artifacts/calibrated_rf_pipeline.joblib")
    calibrated_prob = calibrated.predict_proba(features)[:, 1]
    raw_rf = calibrated.named_steps["classifier"].calibrated_classifiers_[0].estimator.estimator
    raw_prob = raw_rf.predict_proba(calibrated.named_steps["preprocessor"].transform(features))[:, 1]
    calibration = {}
    for name, values in (("RF · raw calibration base", raw_prob), ("RF · calibrated", calibrated_prob)):
        observed, predicted = calibration_curve(labels, values, n_bins=10, strategy="quantile")
        calibration[name] = {"x": predicted.tolist(), "y": observed.tolist()}

    summary = {
        "source": "ULB creditcard.csv; chronological train and validation partitions only",
        "sample": {
            "rows": len(sample),
            "fraud_example_positions": np.flatnonzero(selected["Class"].to_numpy() == 1).tolist(),
            "legitimate_example_positions": np.flatnonzero(selected["Class"].to_numpy() == 0).tolist(),
            "selection": "10 fraud and 90 legitimate validation rows sampled with random_state=42; not representative of class prevalence",
        },
        "eda": {
            "transactions": len(visible),
            "fraud": int(visible["Class"].sum()),
            "median_amount": float(visible["Amount"].median()),
            "class_counts": class_counts.to_dict(orient="records"),
            "hourly": hourly.to_dict(orient="records"),
            "amount_bands": amount.to_dict(orient="records"),
        },
        "model_comparison": load_json_artifact("model_comparison_metrics.json"),
        "calibration_metrics": load_json_artifact("calibration_metrics.json"),
        "threshold_metadata": load_json_artifact("threshold_metadata.json"),
        "validation_pr_curves": curves,
        "validation_fraud_prevalence": float(labels.mean()),
        "validation_calibration_curves": calibration,
    }
    json_path = output_dir / "dashboard_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


if __name__ == "__main__":
    print(build_demo_data())
