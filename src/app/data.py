"""Read-only data and artifact access for the Streamlit dashboard."""

from __future__ import annotations

import json
import csv
import io
from pathlib import Path

import pandas as pd

from src.ingestion.data_loader import EXPECTED_COLUMNS, EXPECTED_ROWS
from src.models.risk_scorer import REQUIRED_COLUMNS, predict_risk, validate_input_schema

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROWS = int(EXPECTED_ROWS * 0.60)
VALIDATION_ROWS = int(EXPECTED_ROWS * 0.20)
VISIBLE_ROWS = TRAIN_ROWS + VALIDATION_ROWS
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 5_000


def load_visible_data(path: Path = ROOT / "data/raw/creditcard.csv") -> pd.DataFrame:
    """Read only the chronological train and validation rows, never the test rows.

    Args:
        path: Original ULB CSV in its chronological order.

    Returns:
        Train and validation records with an explicit partition label.

    Raises:
        ValueError: If the file does not match the expected schema or ordering.
    """
    frame = pd.read_csv(path, nrows=VISIBLE_ROWS)
    if len(frame) != VISIBLE_ROWS or list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Source CSV does not match the expected ULB schema and row count.")
    if not frame["Time"].is_monotonic_increasing:
        raise ValueError("Source CSV must be chronologically ordered for the saved split.")
    frame["partition"] = ["Train"] * TRAIN_ROWS + ["Validation"] * VALIDATION_ROWS
    return frame


def load_json_artifact(name: str, root: Path = ROOT) -> dict | list:
    """Read an existing report artifact without recalculating model metrics."""
    with (root / "artifacts" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_demo_summary(root: Path = ROOT) -> dict:
    """Read the compact train/validation aggregates shipped with the app."""
    with (root / "data/demo/dashboard_summary.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_demo_transactions(root: Path = ROOT) -> pd.DataFrame:
    """Read the public validation-only sample with exactly 30 numeric features."""
    frame = pd.read_csv(root / "data/demo/validation_transactions_100.csv")
    if len(frame) != 100 or list(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError("Demo CSV must have 100 rows and the 30 required features in model order.")
    return validate_batch(frame)


def review_threshold(root: Path = ROOT) -> float:
    """Return the validation selected RF review threshold from its metadata."""
    metadata = load_demo_summary(root)["threshold_metadata"]
    if metadata["model_name"] != "Calibrated RF":
        raise ValueError("Threshold artifact is not for the calibrated Random Forest.")
    return float(metadata["optimal_threshold"])


def validate_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate uploaded transaction features before calling the saved pipeline.

    Args:
        frame: Uploaded CSV content.

    Returns:
        A copy of the original rows, preserving optional identifier columns.
    """
    if frame.empty:
        raise ValueError("CSV has no transaction rows.")
    if frame.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names.")
    reserved = {"fraud_probability", "review_decision", "risk_tier"} & set(frame.columns)
    if reserved:
        raise ValueError(f"CSV contains reserved output columns: {sorted(reserved)}")
    validate_input_schema(frame)
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required feature columns contain missing values.")
    import numpy as np

    if not np.isfinite(frame[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Required feature columns must contain finite numbers.")
    if (frame["Time"] < 0).any() or (frame["Amount"] < 0).any():
        raise ValueError("Time and Amount must be non-negative.")
    return frame.copy()


def parse_batch_csv(payload: bytes) -> pd.DataFrame:
    """Parse a bounded CSV, rejecting duplicate names before pandas mangles them."""
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("CSV exceeds the 5 MiB upload limit.")
    try:
        header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline="")))
    except (StopIteration, UnicodeError, csv.Error) as error:
        raise ValueError("CSV is empty or is not valid UTF-8 text.") from error
    if len(header) != len(set(header)):
        raise ValueError("CSV contains duplicate column names.")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except pd.errors.ParserError as error:
        raise ValueError(f"CSV parsing failed: {error}") from error
    if len(frame) > MAX_UPLOAD_ROWS:
        raise ValueError("CSV exceeds the 5,000-row screening limit.")
    return frame


def score_batch(frame: pd.DataFrame, pipeline: object, threshold: float) -> pd.DataFrame:
    """Score validated rows using the existing calibrated RF risk engine."""
    valid = validate_batch(frame)
    scores = predict_risk(valid, pipeline, threshold=threshold)
    return pd.concat([valid.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
