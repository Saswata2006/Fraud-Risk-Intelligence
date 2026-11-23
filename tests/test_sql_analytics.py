"""Unit tests for DuckDB analytical SQL queries."""

from pathlib import Path
import pandas as pd
import pytest
from src.ingestion.data_loader import load_raw_data
from src.ingestion.sql_analytics import (
    get_duckdb_connection,
    query_transaction_counts_by_class,
    query_average_amount_by_class,
    query_fraud_rate_by_amount_range,
    query_fraud_frequency_by_hour,
    run_all_sql_analytics
)

DATA_PATH = Path("data/raw/creditcard.csv")


@pytest.fixture(scope="session")
def duckdb_con():
    """Fixture providing an in-memory DuckDB connection with transactions table."""
    con = get_duckdb_connection(DATA_PATH)
    yield con
    con.close()


def test_transaction_counts_by_class(duckdb_con):
    """Verify transaction counts by class via DuckDB."""
    res = query_transaction_counts_by_class(duckdb_con)
    assert len(res) == 2
    assert set(res["Class"].tolist()) == {0, 1}
    legit_row = res[res["Class"] == 0].iloc[0]
    fraud_row = res[res["Class"] == 1].iloc[0]
    assert legit_row["transaction_count"] == 284315
    assert fraud_row["transaction_count"] == 492
    assert abs(res["percentage"].sum() - 100.0) < 0.01


def test_average_amount_by_class(duckdb_con):
    """Verify amount statistics by class via DuckDB."""
    res = query_average_amount_by_class(duckdb_con)
    assert len(res) == 2
    legit_row = res[res["Class"] == 0].iloc[0]
    fraud_row = res[res["Class"] == 1].iloc[0]
    assert legit_row["mean_amount"] == 88.29
    assert fraud_row["mean_amount"] == 122.21
    assert legit_row["max_amount"] == 25691.16
    assert fraud_row["max_amount"] == 2125.87


def test_fraud_rate_by_amount_range(duckdb_con):
    """Verify amount tiering logic via DuckDB."""
    res = query_fraud_rate_by_amount_range(duckdb_con)
    assert len(res) == 7
    total_tx = res["total_transactions"].sum()
    assert total_tx == 284807
    total_fraud = res["fraud_count"].sum()
    assert total_fraud == 492


def test_fraud_frequency_by_hour(duckdb_con):
    """Verify elapsed hour bucketing (0 to 47 hours)."""
    res = query_fraud_frequency_by_hour(duckdb_con)
    assert len(res) == 48  # 48 hours in the 172,792 second window
    assert res["total_transactions"].sum() == 284807
    assert res["fraud_count"].sum() == 492


def test_run_all_sql_analytics():
    """Verify end-to-end wrapper returning dictionary of queries."""
    results = run_all_sql_analytics(DATA_PATH)
    assert "class_counts" in results
    assert "amount_stats" in results
    assert "amount_tiers" in results
    assert "hourly_frequency" in results
