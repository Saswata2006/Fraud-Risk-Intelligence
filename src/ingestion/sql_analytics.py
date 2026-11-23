"""DuckDB SQL Analytics Module for Credit Card Fraud Detection.

Leverages DuckDB's vectorized analytical execution to perform SQL-based
profiling, class distribution analysis, amount bucketing, and hourly
temporal aggregation.
"""

from pathlib import Path
from typing import Dict, Union
import duckdb
import pandas as pd


def get_duckdb_connection(
    filepath_or_df: Union[str, Path, pd.DataFrame] = "data/raw/creditcard.csv"
) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection and register the transactions table.

    Args:
        filepath_or_df: Path to CSV or existing Pandas DataFrame.

    Returns:
        duckdb.DuckDBPyConnection with 'transactions' view/table registered.
    """
    con = duckdb.connect(database=":memory:")
    
    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df  # noqa: F841
        con.register("transactions", df)
    else:
        path = Path(filepath_or_df)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found at {path.resolve()}")
        # DuckDB can scan the CSV file directly with vectorized performance
        con.execute(f"CREATE VIEW transactions AS SELECT * FROM read_csv_auto('{path.as_posix()}')")
    
    return con


def query_transaction_counts_by_class(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query total transaction counts, distribution, and volume by class.

    Args:
        con: DuckDB connection.

    Returns:
        pd.DataFrame with Class, label, transaction_count, percentage, total_volume_usd.
    """
    query = """
    SELECT 
        Class,
        CASE WHEN Class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS label,
        COUNT(*) AS transaction_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 4) AS percentage,
        ROUND(SUM(Amount), 2) AS total_volume_usd,
        ROUND(AVG(Amount), 2) AS avg_amount_usd
    FROM transactions
    GROUP BY Class
    ORDER BY Class ASC;
    """
    return con.execute(query).df()


def query_average_amount_by_class(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query detailed descriptive statistics of transaction amounts by class.

    Args:
        con: DuckDB connection.

    Returns:
        pd.DataFrame with statistical metrics (mean, median, std, min, max, percentiles).
    """
    query = """
    SELECT 
        Class,
        CASE WHEN Class = 1 THEN 'Fraudulent' ELSE 'Legitimate' END AS label,
        COUNT(*) AS count,
        ROUND(AVG(Amount), 2) AS mean_amount,
        ROUND(STDDEV_SAMP(Amount), 2) AS std_amount,
        ROUND(MIN(Amount), 2) AS min_amount,
        ROUND(QUANTILE_CONT(Amount, 0.25), 2) AS q25_amount,
        ROUND(MEDIAN(Amount), 2) AS median_amount,
        ROUND(QUANTILE_CONT(Amount, 0.75), 2) AS q75_amount,
        ROUND(QUANTILE_CONT(Amount, 0.95), 2) AS q95_amount,
        ROUND(MAX(Amount), 2) AS max_amount
    FROM transactions
    GROUP BY Class
    ORDER BY Class ASC;
    """
    return con.execute(query).df()


def query_fraud_rate_by_amount_range(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query fraud frequency and fraud rate across defined transaction amount ranges.

    Args:
        con: DuckDB connection.

    Returns:
        pd.DataFrame with amount brackets, total counts, fraud counts, and fraud rates.
    """
    query = """
    WITH tiered_transactions AS (
        SELECT 
            Class,
            Amount,
            CASE 
                WHEN Amount = 0 THEN '1. $0 (Zero-Dollar Auth)'
                WHEN Amount > 0 AND Amount <= 10 THEN '2. $0.01 - $10'
                WHEN Amount > 10 AND Amount <= 50 THEN '3. $10.01 - $50'
                WHEN Amount > 50 AND Amount <= 100 THEN '4. $50.01 - $100'
                WHEN Amount > 100 AND Amount <= 500 THEN '5. $100.01 - $500'
                WHEN Amount > 500 AND Amount <= 1000 THEN '6. $500.01 - $1,000'
                ELSE '7. > $1,000'
            END AS amount_tier,
            CASE 
                WHEN Amount = 0 THEN 1
                WHEN Amount > 0 AND Amount <= 10 THEN 2
                WHEN Amount > 10 AND Amount <= 50 THEN 3
                WHEN Amount > 50 AND Amount <= 100 THEN 4
                WHEN Amount > 100 AND Amount <= 500 THEN 5
                WHEN Amount > 500 AND Amount <= 1000 THEN 6
                ELSE 7
            END AS tier_order
        FROM transactions
    )
    SELECT 
        amount_tier,
        COUNT(*) AS total_transactions,
        SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legit_count,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
        ROUND(SUM(CASE WHEN Class = 1 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0.0 END), 2) AS total_fraud_amount_usd
    FROM tiered_transactions
    GROUP BY amount_tier, tier_order
    ORDER BY tier_order ASC;
    """
    return con.execute(query).df()


def query_fraud_frequency_by_hour(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Query transaction volume and fraud frequency bucketed by elapsed hour.

    Args:
        con: DuckDB connection.

    Returns:
        pd.DataFrame aggregated by elapsed hour (0 to 47).
    """
    query = """
    SELECT 
        CAST(FLOOR(Time / 3600) AS INTEGER) AS elapsed_hour,
        (CAST(FLOOR(Time / 3600) AS INTEGER) % 24) AS hour_of_day,
        COUNT(*) AS total_transactions,
        SUM(CASE WHEN Class = 0 THEN 1 ELSE 0 END) AS legit_count,
        SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraud_count,
        ROUND(SUM(CASE WHEN Class = 1 THEN 1.0 ELSE 0.0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
        ROUND(AVG(Amount), 2) AS avg_amount_usd
    FROM transactions
    GROUP BY elapsed_hour, hour_of_day
    ORDER BY elapsed_hour ASC;
    """
    return con.execute(query).df()


def run_all_sql_analytics(
    filepath_or_df: Union[str, Path, pd.DataFrame] = "data/raw/creditcard.csv"
) -> Dict[str, pd.DataFrame]:
    """Execute all SQL analytics queries and return a dictionary of result DataFrames.

    Args:
        filepath_or_df: Path to CSV or existing DataFrame.

    Returns:
        Dictionary mapping query name to resulting pd.DataFrame.
    """
    con = get_duckdb_connection(filepath_or_df)
    results = {
        "class_counts": query_transaction_counts_by_class(con),
        "amount_stats": query_average_amount_by_class(con),
        "amount_tiers": query_fraud_rate_by_amount_range(con),
        "hourly_frequency": query_fraud_frequency_by_hour(con),
    }
    con.close()
    return results
