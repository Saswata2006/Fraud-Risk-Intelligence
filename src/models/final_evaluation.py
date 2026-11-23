"""Record a single evaluation of the frozen calibrated RF on the final holdout."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn import metrics

from src.features.data_split import create_chronological_splits
from src.ingestion.data_loader import load_raw_data
from src.models.calibration import load_calibrated_pipeline
from src.models.risk_scorer import predict_risk


def main() -> None:
    """Verify the frozen artifacts and write aggregate metrics once."""
    root = Path(__file__).resolve().parents[2]
    report = root / "reports/final_test_metrics.json"
    if report.exists():
        raise SystemExit("Final report already exists.")
    artifact = root / "artifacts/calibrated_rf_pipeline.joblib"
    policy = json.loads((root / "artifacts/threshold_metadata.json").read_text())
    bundle = json.loads((root / "data/demo/dashboard_summary.json").read_text())
    assert policy == bundle["threshold_metadata"]
    threshold = policy["optimal_threshold"]
    assert policy["model_name"] == "Calibrated RF" and policy["selection_objective"] == "f1"
    assert abs(threshold - 0.390) < 1e-12
    pipeline = load_calibrated_pipeline(artifact)
    assert pipeline.named_steps["classifier"].method == "sigmoid"
    print(pipeline.named_steps["preprocessor"], pipeline.named_steps["classifier"].estimator)
    train, validation, test = create_chronological_splits(load_raw_data(root / "data/raw/creditcard.csv"))
    assert (len(train), len(validation), len(test)) == (170884, 56961, 56962)
    assert train.Time.max() <= validation.Time.min() <= validation.Time.max() <= test.Time.min()
    result = {"model_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "threshold": threshold}
    with report.open("x") as output:
        for name, frame in (("validation", validation), ("final_test", test)):
            y = frame["Class"]
            scored = predict_risk(frame.drop(columns="Class"), pipeline, threshold)
            p, decision = scored.fraud_probability, scored.review_decision
            tn, fp, fn, tp = map(int, metrics.confusion_matrix(y, decision, labels=[0, 1]).ravel())
            precision, recall, _ = metrics.precision_recall_curve(y, p)
            cost = policy["cost_assumptions"]
            result[name] = {
                "rows": len(frame), "positives": tp + fn,
                "average_precision": metrics.average_precision_score(y, p),
                "roc_auc": metrics.roc_auc_score(y, p),
                "precision": metrics.precision_score(y, decision),
                "recall": metrics.recall_score(y, decision),
                "f1": metrics.f1_score(y, decision),
                "brier": metrics.brier_score_loss(y, p), "log_loss": metrics.log_loss(y, p),
                "confusion_matrix": [[tn, fp], [fn, tp]], "false_positives": fp, "false_negatives": fn,
                "max_recall_at_precision_0.80": float(np.max(recall[precision >= .80])),
                "illustrative_loss_reduction_usd_vs_no_reviews": tp * cost["fn_cost_usd"] - fp * cost["fp_cost_usd"],
            }
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
