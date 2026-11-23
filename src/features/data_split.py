"""Chronological data splitting and cross-partition leakage audit module.

Splits transactions chronologically by elapsed time into train (60%),
validation (20%), and test (20%) partitions, while auditing exact feature
duplicates across partitions to prevent cross-partition leakage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class SplitReport:
    """Summary of data split statistics and cross-partition duplicate audit."""
    train_count: int
    val_count: int
    test_count: int
    train_fraud_count: int
    val_fraud_count: int
    test_fraud_count: int
    train_fraud_rate: float
    val_fraud_rate: float
    test_fraud_rate: float
    full_row_cross_train_val: int
    full_row_cross_train_test: int
    full_row_cross_val_test: int
    feature_profile_cross_train_val: int
    feature_profile_cross_train_test: int
    feature_profile_cross_val_test: int
    val_leaked_fraud_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return {
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "train_fraud_count": self.train_fraud_count,
            "val_fraud_count": self.val_fraud_count,
            "test_fraud_count": self.test_fraud_count,
            "train_fraud_rate": self.train_fraud_rate,
            "val_fraud_rate": self.val_fraud_rate,
            "test_fraud_rate": self.test_fraud_rate,
            "full_row_cross_train_val": self.full_row_cross_train_val,
            "full_row_cross_train_test": self.full_row_cross_train_test,
            "full_row_cross_val_test": self.full_row_cross_val_test,
            "feature_profile_cross_train_val": self.feature_profile_cross_train_val,
            "feature_profile_cross_train_test": self.feature_profile_cross_train_test,
            "feature_profile_cross_val_test": self.feature_profile_cross_val_test,
            "val_leaked_fraud_count": self.val_leaked_fraud_count,
        }


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    time_col: str = "Time",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort transactions chronologically and partition into train, val, test.

    Args:
        df: Input DataFrame containing credit card transactions.
        train_ratio: Proportion of transactions in the training partition (default 0.60).
        val_ratio: Proportion of transactions in the validation partition (default 0.20).
        test_ratio: Proportion of transactions in the test partition (default 0.20).
        time_col: Column name representing elapsed transaction time.

    Returns:
        Tuple of (train_df, val_df, test_df) reset to fresh 0-based indices.

    Raises:
        ValueError: If partition ratios do not sum to 1.0 or time_col is missing.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0: got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}"
        )
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in DataFrame")

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def audit_cross_partition_duplicates(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Class",
    time_col: str = "Time",
) -> SplitReport:
    """Audit exact record duplicates and feature profile duplicates across partitions.

    Args:
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.
        target_col: Target column name.
        time_col: Time column name.

    Returns:
        SplitReport dataclass detailing sample counts, fraud rates, and cross-partition overlap.
    """
    train_fraud = int(train_df[target_col].sum()) if target_col in train_df else 0
    val_fraud = int(val_df[target_col].sum()) if target_col in val_df else 0
    test_fraud = int(test_df[target_col].sum()) if target_col in test_df else 0

    train_rate = (train_fraud / len(train_df)) * 100 if len(train_df) > 0 else 0.0
    val_rate = (val_fraud / len(val_df)) * 100 if len(val_df) > 0 else 0.0
    test_rate = (test_fraud / len(test_df)) * 100 if len(test_df) > 0 else 0.0

    # 1. Full-row duplicate audit (all columns including Time and Class)
    full_train = set(tuple(x) for x in train_df.values)
    full_val = set(tuple(x) for x in val_df.values)
    full_test = set(tuple(x) for x in test_df.values)

    cross_full_tv = len(full_train.intersection(full_val))
    cross_full_tt = len(full_train.intersection(full_test))
    cross_full_vt = len(full_val.intersection(full_test))

    # 2. Feature-profile duplicate audit (V1-V28 and Amount, excluding Time and Class)
    feat_cols = [c for c in train_df.columns if c not in [target_col, time_col]]
    v_train = set(tuple(x) for x in train_df[feat_cols].values)
    v_val = set(tuple(x) for x in val_df[feat_cols].values)
    v_test = set(tuple(x) for x in test_df[feat_cols].values)

    cross_v_tv = len(v_train.intersection(v_val))
    cross_v_tt = len(v_train.intersection(v_test))
    cross_v_vt = len(v_val.intersection(v_test))

    # Check if any validation rows matching training feature profiles are fraudulent
    val_tuples = val_df[feat_cols].apply(tuple, axis=1)
    val_leak_mask = val_tuples.isin(v_train)
    val_leaked_fraud = int(val_df.loc[val_leak_mask, target_col].sum()) if target_col in val_df else 0

    return SplitReport(
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        train_fraud_count=train_fraud,
        val_fraud_count=val_fraud,
        test_fraud_count=test_fraud,
        train_fraud_rate=round(train_rate, 4),
        val_fraud_rate=round(val_rate, 4),
        test_fraud_rate=round(test_rate, 4),
        full_row_cross_train_val=cross_full_tv,
        full_row_cross_train_test=cross_full_tt,
        full_row_cross_val_test=cross_full_vt,
        feature_profile_cross_train_val=cross_v_tv,
        feature_profile_cross_train_test=cross_v_tt,
        feature_profile_cross_val_test=cross_v_vt,
        val_leaked_fraud_count=val_leaked_fraud,
    )


def filter_cross_partition_duplicates(
    eval_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Class",
    time_col: str = "Time",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Isolate evaluation rows that share identical feature profiles with reference partition.

    Args:
        eval_df: Evaluation partition (e.g. validation).
        reference_df: Reference partition (e.g. train).
        feature_cols: Columns defining the feature profile. If None, uses all except target and time.
        target_col: Target column name.
        time_col: Time column name.

    Returns:
        Tuple of (clean_eval_df, leaked_eval_df) for comparative evaluation.
    """
    if feature_cols is None:
        feature_cols = [c for c in eval_df.columns if c not in [target_col, time_col]]

    ref_tuples = set(tuple(x) for x in reference_df[feature_cols].values)
    eval_tuples = eval_df[feature_cols].apply(tuple, axis=1)
    is_overlap = eval_tuples.isin(ref_tuples)

    clean_eval_df = eval_df.loc[~is_overlap].copy().reset_index(drop=True)
    leaked_eval_df = eval_df.loc[is_overlap].copy().reset_index(drop=True)

    return clean_eval_df, leaked_eval_df
