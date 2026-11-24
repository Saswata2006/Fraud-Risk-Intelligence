"""Focused checks for dashboard data boundaries, scoring, and chart construction."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.app.charts import calibration_figure, precision_recall_figure, shap_contribution_figure
from src.app.data import (
    MAX_UPLOAD_BYTES, MAX_UPLOAD_ROWS, TRAIN_ROWS, VALIDATION_ROWS, build_review_queue,
    load_demo_summary, load_demo_transactions, load_visible_data, parse_batch_csv,
    review_threshold, score_batch, validate_batch,
)
from src.models.risk_scorer import REQUIRED_COLUMNS


@pytest.mark.dataset
def test_visible_data_excludes_final_test_rows():
    """Only the train and validation prefix is loaded for the dashboard."""
    data = load_visible_data()
    assert len(data) == TRAIN_ROWS + VALIDATION_ROWS
    assert (data.iloc[:TRAIN_ROWS]["partition"] == "Train").all()
    assert (data.iloc[TRAIN_ROWS:]["partition"] == "Validation").all()
    assert data["Time"].is_monotonic_increasing


def test_review_threshold_comes_from_rf_artifact():
    """The review cutoff is loaded from saved validation metadata."""
    assert review_threshold() == pytest.approx(0.39)


@pytest.fixture
def batch() -> pd.DataFrame:
    """Create raw feature rows without using the withheld test set."""
    frame = pd.DataFrame(np.zeros((2, len(REQUIRED_COLUMNS))), columns=REQUIRED_COLUMNS)
    frame.insert(0, "transaction_id", ["a", "b"])
    return frame


def test_batch_scores_and_preserves_identifiers(batch: pd.DataFrame):
    """Scoring calls the shared risk engine with a distinct review cutoff."""
    pipeline = Mock()
    pipeline.predict_proba.return_value = np.array([[0.65, 0.35], [0.40, 0.60]])
    result = score_batch(batch, pipeline, threshold=0.39)
    assert result["transaction_id"].tolist() == ["a", "b"]
    assert result["review_decision"].tolist() == [False, True]
    assert result["risk_tier"].tolist() == ["Medium", "High"]
    pipeline.predict_proba.assert_called_once()


def test_batch_passes_model_features_in_required_order(batch: pd.DataFrame):
    """Reordered upload columns cannot change the pipeline's feature order."""
    pipeline = Mock()
    pipeline.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    shuffled = batch[["transaction_id", *reversed(REQUIRED_COLUMNS)]]
    score_batch(shuffled, pipeline, threshold=0.39)
    model_input = pipeline.predict_proba.call_args.args[0]
    assert model_input.columns.tolist() == REQUIRED_COLUMNS


@pytest.mark.dataset
def test_demo_csv_is_validation_only_and_unlabeled():
    """The published 100-row CSV matches sampled validation feature rows."""
    sample = load_demo_transactions()
    validation = load_visible_data().iloc[TRAIN_ROWS:]
    assert sample.shape == (100, 30)
    assert sample.columns.tolist() == REQUIRED_COLUMNS
    joined = sample.merge(validation[REQUIRED_COLUMNS], on=REQUIRED_COLUMNS, how="left", indicator=True)
    assert (joined["_merge"] == "both").all()
    metadata = load_demo_summary()["sample"]
    assert len(metadata["fraud_example_positions"]) == 10
    assert len(metadata["legitimate_example_positions"]) == 90


def test_demo_csv_scores_and_export_has_all_decision_columns():
    """Round-trip the public CSV through the saved RF and scored CSV export."""
    from io import StringIO

    from src.models.calibration import load_calibrated_pipeline

    root = Path(__file__).resolve().parents[1]
    uploaded = parse_batch_csv((root / "data/demo/validation_transactions_100.csv").read_bytes())
    pipeline = load_calibrated_pipeline(root / "artifacts/calibrated_rf_pipeline.joblib")
    scored = score_batch(uploaded, pipeline, review_threshold())
    exported = pd.read_csv(StringIO(scored.to_csv(index=False)))
    assert len(exported) == 100
    assert exported.columns.tolist() == [*REQUIRED_COLUMNS, "fraud_probability", "review_decision", "risk_tier"]
    assert exported["fraud_probability"].between(0, 1).all()
    assert exported["review_decision"].isin([True, False]).all()
    assert exported["risk_tier"].isin(["Low", "Medium", "High"]).all()


@pytest.mark.dataset
def test_dashboard_summary_excludes_test_partition():
    """Public aggregates match the train/validation prefix and contain no test rows."""
    summary = load_demo_summary()
    visible = load_visible_data()
    assert summary["eda"]["transactions"] == len(visible) == TRAIN_ROWS + VALIDATION_ROWS
    assert summary["eda"]["fraud"] == int(visible["Class"].sum())
    assert sum(row["Transactions"] for row in summary["eda"]["class_counts"]) == len(visible)
    assert {row["partition"] for row in summary["eda"]["class_counts"]} == {"Train", "Validation"}


def test_uploaded_csv_rejects_duplicate_header(batch: pd.DataFrame):
    """Duplicate headers are rejected before pandas renames one with a suffix."""
    payload = batch.to_csv(index=False).encode("utf-8")
    payload = payload.replace(b"V1,V2", b"V1,V1")
    with pytest.raises(ValueError, match="duplicate column names"):
        parse_batch_csv(payload)


def test_upload_size_boundary(batch, monkeypatch):
    payload = batch.to_csv(index=False).encode()
    payload = payload.replace(b"\na,", b"\na" + b"x" * (MAX_UPLOAD_BYTES - len(payload)) + b",", 1)
    assert len(payload) == MAX_UPLOAD_BYTES
    assert len(validate_batch(parse_batch_csv(payload))) == len(batch)
    monkeypatch.setattr(pd, "read_csv", Mock(side_effect=AssertionError("unbounded CSV parsed")))
    with pytest.raises(ValueError, match="5 MiB"):
        parse_batch_csv(payload + b"\n")


def test_upload_row_boundary_counts_logical_records(batch, monkeypatch):
    record = batch.iloc[[0]].assign(transaction_id="test,\n001")
    header, rows = record.to_csv(index=False).encode().split(b"\n", 1)
    payload = header + b"\n" + rows * MAX_UPLOAD_ROWS
    parsed = validate_batch(parse_batch_csv(payload))
    assert len(parsed) == MAX_UPLOAD_ROWS
    assert parsed.iloc[-1]["transaction_id"] == "test,\n001"
    monkeypatch.setattr(pd, "read_csv", Mock(side_effect=AssertionError("unbounded CSV parsed")))
    with pytest.raises(ValueError, match="5,000-row"):
        parse_batch_csv(payload + rows)


@pytest.mark.parametrize("payload", [b'id,Time\n"unterminated', b"id,Time\n1,2,3", b"Time\n\xff"])
def test_malformed_upload_is_rejected(payload):
    with pytest.raises(ValueError):
        parse_batch_csv(payload)


def test_review_queue_preserves_csv_identity_through_export(batch):
    from io import StringIO

    frame = pd.concat([batch, batch], ignore_index=True)
    frame["transaction_id"] = ["0001", "NA", "0001", "1e6"]
    frame["input_row"] = ["A", "B", "C", "D"]
    uploaded = parse_batch_csv(frame.to_csv(index=False).encode())
    pipeline = Mock()
    pipeline.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8], [0.2, 0.8], [0.6, 0.4]])
    scored = score_batch(uploaded, pipeline, review_threshold())
    snapshot = scored.copy()
    queue = build_review_queue(scored, flagged_only=True, top_k=2)
    exported = pd.read_csv(StringIO(queue.to_csv(index=False)), dtype=str, keep_default_na=False)
    assert exported["_input_row"].tolist() == ["2", "3"]
    assert exported["input_row"].tolist() == ["B", "C"]
    assert exported["transaction_id"].tolist() == ["NA", "0001"]
    assert exported["review_decision"].tolist() == ["True", "True"]
    assert exported["risk_tier"].tolist() == ["High", "High"]
    assert exported["fraud_probability"].tolist() == ["0.8", "0.8"]
    assert build_review_queue(scored, flagged_only=False, top_k=10)["_input_row"].tolist() == [2, 3, 4, 1]
    flagged = build_review_queue(scored, flagged_only=True, top_k=10)
    assert flagged["risk_tier"].tolist() == ["High", "High", "Medium"]
    empty = build_review_queue(scored.iloc[:1], flagged_only=True, top_k=1)
    assert empty.empty and empty.columns.equals(queue.columns)
    pd.testing.assert_frame_equal(scored, snapshot)
    pipeline.predict_proba.assert_called_once()


@pytest.mark.parametrize("change", ["missing", "null", "text", "infinite", "negative_amount", "reserved"])
def test_batch_rejects_invalid_features(batch: pd.DataFrame, change: str):
    """Malformed CSV rows fail before any model call."""
    if change == "missing":
        batch = batch.drop(columns="V28")
    elif change == "null":
        batch.loc[0, "V1"] = np.nan
    elif change == "text":
        batch["V1"] = "bad"
    elif change == "infinite":
        batch.loc[0, "V1"] = np.inf
    elif change == "negative_amount":
        batch.loc[0, "Amount"] = -1
    else:
        batch["risk_tier"] = "High"
    with pytest.raises(ValueError):
        validate_batch(batch)


def test_plotly_diagnostics_use_supplied_values():
    """Figures reflect validation arrays supplied to the chart functions."""
    labels = pd.Series([0, 0, 1, 1])
    scores = {"RF": np.array([0.01, 0.10, 0.60, 0.90])}
    assert len(precision_recall_figure(labels, scores).data) == 1
    assert len(calibration_figure(labels, scores).data) == 2
    assert len(shap_contribution_figure(np.array([0.2, -0.1]), ["V1", "V2"]).data) == 1


def test_all_dashboard_pages_render_without_raw_csv(monkeypatch: pytest.MonkeyPatch):
    """Exercise every page while any attempt to read the raw CSV fails."""
    from streamlit.testing.v1 import AppTest

    original_read_csv = pd.read_csv

    def guarded_read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if "data/raw/creditcard.csv" in str(path):
            raise AssertionError("Dashboard accessed the ignored raw CSV")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/app/dashboard.py", default_timeout=120).run()
    assert app.title[0].value == "Data Intelligence"
    assert not app.error and not app.exception
    for page in ("Model Performance", "Fraud Prediction", "Batch Screening"):
        app.sidebar.radio[0].set_value(page).run()
        assert app.title[0].value == page
        assert not app.error and not app.exception


def test_shap_renders_for_validation_fraud_and_legitimate_samples():
    """On-demand TreeSHAP charts render for both known validation classes."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "src/app/dashboard.py", default_timeout=120).run()
    app.sidebar.radio[0].set_value("Fraud Prediction").run()
    for sample_type in ("Validation fraud examples", "Validation legitimate examples"):
        app.radio[0].set_value(sample_type).run()
        app.button[0].click().run()
        assert not app.error and not app.exception
        assert len(app.get("plotly_chart")) == 1


def test_batch_page_filters_and_downloads_the_displayed_queue(monkeypatch):
    from io import StringIO

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    downloads = {}
    original_download = st.download_button

    def record_download(label, data, file_name, *args, **kwargs):
        downloads[file_name] = data
        return original_download(label, data, file_name, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", record_download)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / "src/app/dashboard.py", default_timeout=120).run()
    app.sidebar.radio[0].set_value("Batch Screening").run()
    payload = (root / "data/demo/validation_transactions_100.csv").read_bytes()
    app.file_uploader[0].upload("demo.csv", payload, "text/csv").run()
    assert not app.error and not app.exception
    full = pd.read_csv(StringIO(downloads["fraud_screening_results.csv"]))
    flags = full["review_decision"].sum()
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Total scored"] == "100"
    assert metrics["Review flags"] == str(flags)
    assert metrics["Review rate"] == f"{flags / len(full):.1%}"
    assert 0 < flags < len(full)
    queue = app.dataframe[0].value
    assert queue["review_decision"].all() and len(queue) == min(25, flags)

    app.checkbox[0].uncheck().run()
    app.number_input[0].set_value(2).run()
    assert not app.error and not app.exception
    assert metrics == {m.label: m.value for m in app.metric}
    top = pd.read_csv(StringIO(downloads["fraud_review_queue.csv"]))
    pd.testing.assert_frame_equal(top, app.dataframe[0].value.reset_index(drop=True))
    expected = full.sort_values("fraud_probability", ascending=False, kind="stable").head(2)
    assert top["input_row"].tolist() == (expected.index + 1).tolist()
    assert len(pd.read_csv(StringIO(downloads["fraud_screening_results.csv"]))) == 100

    # Replacing the upload must discard the previous batch, even for the same filename.
    single = full.loc[~full["review_decision"], REQUIRED_COLUMNS].head(1)
    app.file_uploader[0].set_value(("demo.csv", single.to_csv(index=False).encode(), "text/csv")).run()
    app.checkbox[0].check().run()
    assert not app.error and not app.exception
    assert {m.label: m.value for m in app.metric}["Total scored"] == "1"
    empty = pd.read_csv(StringIO(downloads["fraud_review_queue.csv"]))
    assert empty.empty and empty.columns.tolist() == top.columns.tolist()
    assert len(pd.read_csv(StringIO(downloads["fraud_screening_results.csv"]))) == 1
