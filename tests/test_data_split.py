"""Unit tests for chronological data splitting and duplicate leakage audit."""

from pathlib import Path
import pandas as pd
import pytest

pytestmark = pytest.mark.dataset
from src.ingestion.data_loader import load_raw_data
from src.features.data_split import (
    create_chronological_splits,
    audit_cross_partition_duplicates,
    filter_cross_partition_duplicates,
)

DATA_PATH = Path("data/raw/creditcard.csv")


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Fixture providing raw data."""
    return load_raw_data(DATA_PATH)


def test_chronological_split_dimensions(raw_df: pd.DataFrame):
    """Verify split ratios, row conservation, and chronological order."""
    train_df, val_df, test_df = create_chronological_splits(raw_df, 0.60, 0.20, 0.20)

    # 1. Row conservation
    assert len(train_df) + len(val_df) + len(test_df) == len(raw_df)
    assert len(train_df) == 170884
    assert len(val_df) == 56961
    assert len(test_df) == 56962

    # 2. Strict chronological ordering across boundaries
    assert train_df["Time"].max() <= val_df["Time"].min()
    assert val_df["Time"].max() <= test_df["Time"].min()

    # 3. Target class conservation
    total_fraud = raw_df["Class"].sum()
    split_fraud = train_df["Class"].sum() + val_df["Class"].sum() + test_df["Class"].sum()
    assert total_fraud == split_fraud == 492


def test_split_ratio_validation(raw_df: pd.DataFrame):
    """Verify ValueError is raised if split ratios do not sum to 1.0."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        create_chronological_splits(raw_df, 0.50, 0.20, 0.20)


def test_audit_cross_partition_duplicates(raw_df: pd.DataFrame):
    """Verify duplicate audit report metrics."""
    train_df, val_df, test_df = create_chronological_splits(raw_df, 0.60, 0.20, 0.20)
    report = audit_cross_partition_duplicates(train_df, val_df, test_df)

    # Full row duplicates across partitions must be 0
    assert report.full_row_cross_train_val == 0
    assert report.full_row_cross_train_test == 0
    assert report.full_row_cross_val_test == 0

    # Fraud sample counts
    assert report.train_fraud_count == 360
    assert report.val_fraud_count == 57
    assert report.test_fraud_count == 75

    # Leaked fraud count in validation must be 0
    assert report.val_leaked_fraud_count == 0


def test_filter_cross_partition_duplicates(raw_df: pd.DataFrame):
    """Verify filter_cross_partition_duplicates separates overlap correctly."""
    train_df, val_df, _ = create_chronological_splits(raw_df, 0.60, 0.20, 0.20)
    clean_val, leaked_val = filter_cross_partition_duplicates(val_df, train_df)

    assert len(clean_val) + len(leaked_val) == len(val_df)
    assert leaked_val["Class"].sum() == 0  # No leaked fraud cases
    assert len(leaked_val) == 561
