"""Data ingestion and validation module for Credit Card Fraud Detection.

Provides reproducible functions to load, validate, and inspect the
Kaggle ULB credit card transaction dataset with strict schema and
data-quality checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


EXPECTED_ROWS = 284807
EXPECTED_COLS = 31
EXPECTED_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount", "Class"
]


@dataclass
class ValidationReport:
    """Summary of data ingestion validation checks."""
    is_valid: bool
    total_rows: int
    total_cols: int
    missing_values_count: int
    duplicate_rows_count: int
    duplicate_fraud_count: int
    duplicate_legit_count: int
    zero_amount_count: int
    zero_amount_fraud_count: int
    negative_amount_count: int
    fraud_count: int
    legit_count: int
    fraud_percentage: float
    column_names: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_cols": self.total_cols,
            "missing_values_count": self.missing_values_count,
            "duplicate_rows_count": self.duplicate_rows_count,
            "duplicate_fraud_count": self.duplicate_fraud_count,
            "duplicate_legit_count": self.duplicate_legit_count,
            "zero_amount_count": self.zero_amount_count,
            "zero_amount_fraud_count": self.zero_amount_fraud_count,
            "negative_amount_count": self.negative_amount_count,
            "fraud_count": self.fraud_count,
            "legit_count": self.legit_count,
            "fraud_percentage": self.fraud_percentage,
            "column_count": len(self.column_names),
            "errors": self.errors,
        }


def load_raw_data(filepath: str | Path = "data/raw/creditcard.csv") -> pd.DataFrame:
    """Load the raw credit card transactions CSV file.

    Args:
        filepath: Path to creditcard.csv.

    Returns:
        pd.DataFrame containing 284,807 transactions.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If file cannot be read as a DataFrame.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found at {path.resolve()}")
    
    df = pd.read_csv(path)
    return df


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate dataset structure, types, missing values, duplicates, and ranges.

    Args:
        df: Pandas DataFrame of credit card transactions.

    Returns:
        ValidationReport dataclass containing audit findings and pass/fail flag.
    """
    errors: List[str] = []

    # 1. Dimension validation
    rows, cols = df.shape
    if rows != EXPECTED_ROWS:
        errors.append(f"Row count mismatch: expected {EXPECTED_ROWS}, found {rows}")
    if cols != EXPECTED_COLS:
        errors.append(f"Column count mismatch: expected {EXPECTED_COLS}, found {cols}")

    # 2. Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(list(missing_cols))}")

    # 3. Missing values check
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        errors.append(f"Found {null_count} missing values in dataset")

    # 4. Duplicate checks (preserving first occurrence)
    duplicates_mask = df.duplicated(keep="first")
    duplicate_rows_count = int(duplicates_mask.sum())
    
    if "Class" in df.columns:
        duplicate_fraud_count = int(df.loc[duplicates_mask, "Class"].sum())
        duplicate_legit_count = duplicate_rows_count - duplicate_fraud_count
        fraud_count = int((df["Class"] == 1).sum())
        legit_count = int((df["Class"] == 0).sum())
        fraud_percentage = round((fraud_count / rows) * 100, 4) if rows > 0 else 0.0
    else:
        duplicate_fraud_count = 0
        duplicate_legit_count = duplicate_rows_count
        fraud_count = 0
        legit_count = 0
        fraud_percentage = 0.0

    # 5. Amount validation
    if "Amount" in df.columns:
        zero_mask = df["Amount"] == 0.0
        zero_amount_count = int(zero_mask.sum())
        zero_amount_fraud_count = int(df.loc[zero_mask, "Class"].sum()) if "Class" in df.columns else 0
        negative_amount_count = int((df["Amount"] < 0.0).sum())
        if negative_amount_count > 0:
            errors.append(f"Found {negative_amount_count} transactions with negative amounts")
    else:
        zero_amount_count = 0
        zero_amount_fraud_count = 0
        negative_amount_count = 0
        errors.append("Column 'Amount' missing from dataset")

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_rows=rows,
        total_cols=cols,
        missing_values_count=null_count,
        duplicate_rows_count=duplicate_rows_count,
        duplicate_fraud_count=duplicate_fraud_count,
        duplicate_legit_count=duplicate_legit_count,
        zero_amount_count=zero_amount_count,
        zero_amount_fraud_count=zero_amount_fraud_count,
        negative_amount_count=negative_amount_count,
        fraud_count=fraud_count,
        legit_count=legit_count,
        fraud_percentage=fraud_percentage,
        column_names=list(df.columns),
        errors=errors,
    )
