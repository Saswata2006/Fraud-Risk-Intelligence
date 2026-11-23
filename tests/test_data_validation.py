"""Unit tests for data validation and schema integrity."""

from pathlib import Path
import pandas as pd
import pytest
from src.ingestion.data_loader import (
    load_raw_data,
    validate_dataset,
    EXPECTED_ROWS,
    EXPECTED_COLS,
    EXPECTED_COLUMNS
)

DATA_PATH = Path("data/raw/creditcard.csv")


@pytest.fixture(scope="session")
def dataset_df() -> pd.DataFrame:
    """Fixture to load dataset once for session."""
    assert DATA_PATH.exists(), f"Missing dataset file at {DATA_PATH}"
    return load_raw_data(DATA_PATH)


def test_dataset_dimensions(dataset_df: pd.DataFrame):
    """Verify exact row and column dimensions."""
    assert dataset_df.shape == (EXPECTED_ROWS, EXPECTED_COLS)


def test_dataset_columns(dataset_df: pd.DataFrame):
    """Verify presence and order of all expected column headers."""
    assert list(dataset_df.columns) == EXPECTED_COLUMNS


def test_missing_values(dataset_df: pd.DataFrame):
    """Verify dataset contains zero missing/null values."""
    assert dataset_df.isnull().sum().sum() == 0


def test_validation_report(dataset_df: pd.DataFrame):
    """Verify validation report flags and counts."""
    report = validate_dataset(dataset_df)
    assert report.is_valid is True
    assert report.total_rows == 284807
    assert report.total_cols == 31
    assert report.missing_values_count == 0
    assert report.negative_amount_count == 0
    assert report.fraud_count == 492
    assert report.legit_count == 284315
    assert report.duplicate_rows_count == 1081
    assert report.duplicate_fraud_count == 19
    assert report.zero_amount_count == 1825
    assert report.zero_amount_fraud_count == 27
