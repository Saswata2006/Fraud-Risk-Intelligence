"""Unit tests for Random Forest and XGBoost pipelines, benchmarking, and serialization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.ingestion.data_loader import load_raw_data
from src.features.data_split import create_chronological_splits
from src.models.random_forest import (
    build_random_forest_pipeline,
    load_rf_model,
    get_rf_feature_importances,
)
from src.models.xgboost_model import (
    build_xgboost_pipeline,
    load_xgb_model,
    compute_scale_pos_weight,
    get_xgb_feature_importances,
)
from src.models.model_comparison import benchmark_models

DATA_PATH = Path("data/raw/creditcard.csv")
RF_MODEL_PATH = Path("artifacts/random_forest_pipeline.joblib")
XGB_MODEL_PATH = Path("artifacts/xgboost_pipeline.joblib")


@pytest.fixture(scope="session")
def partitions():
    """Fixture providing chronological splits for tree testing."""
    df = load_raw_data(DATA_PATH)
    train_df, val_df, test_df = create_chronological_splits(df)
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]
    return X_train, y_train, X_val, y_val


def test_consistent_dataset_partitions(partitions):
    """Verify that partitions maintain exact shape and target proportions."""
    X_train, y_train, X_val, y_val = partitions
    assert len(X_train) == 170884
    assert len(X_val) == 56961
    assert y_train.sum() == 360
    assert y_val.sum() == 57


def test_scale_pos_weight_calculation(partitions):
    """Verify training-only scale_pos_weight calculation."""
    _, y_train, _, _ = partitions
    spw = compute_scale_pos_weight(y_train)
    expected_spw = (len(y_train) - y_train.sum()) / y_train.sum()
    assert abs(spw - expected_spw) < 1e-4
    assert spw > 400.0  # ~473.68


def test_random_forest_pipeline_predictions_and_serialization(partitions):
    """Verify Random Forest pipeline output shapes, probabilities, and reloadability."""
    _, _, X_val, y_val = partitions
    assert RF_MODEL_PATH.exists(), f"Missing artifact at {RF_MODEL_PATH}"
    pipeline = load_rf_model(RF_MODEL_PATH)

    preds = pipeline.predict(X_val)
    probs = pipeline.predict_proba(X_val)

    # 1. Output shapes
    assert preds.shape == (len(X_val),)
    assert probs.shape == (len(X_val), 2)

    # 2. Probability bounds
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # 3. Feature importances sum to 1.0
    imp_df = get_rf_feature_importances(pipeline)
    assert len(imp_df) == 30
    assert abs(imp_df["importance"].sum() - 1.0) < 1e-4


def test_xgboost_pipeline_predictions_and_serialization(partitions):
    """Verify XGBoost pipeline output shapes, probabilities, and reloadability."""
    _, _, X_val, y_val = partitions
    assert XGB_MODEL_PATH.exists(), f"Missing artifact at {XGB_MODEL_PATH}"
    pipeline = load_xgb_model(XGB_MODEL_PATH)

    preds = pipeline.predict(X_val)
    probs = pipeline.predict_proba(X_val)

    # 1. Output shapes
    assert preds.shape == (len(X_val),)
    assert probs.shape == (len(X_val), 2)

    # 2. Probability bounds
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # 3. Feature importances extracted properly
    imp_df = get_xgb_feature_importances(pipeline)
    assert len(imp_df) == 30
    assert (imp_df["importance"] >= 0.0).all()


def test_benchmark_models_comparison_logic(partitions):
    """Verify benchmarking function produces valid comparison table."""
    _, _, X_val, y_val = partitions
    rf_pipe = load_rf_model(RF_MODEL_PATH)
    xgb_pipe = load_xgb_model(XGB_MODEL_PATH)

    comp_df = benchmark_models(
        {"RF": rf_pipe, "XGB": xgb_pipe},
        X_val,
        y_val,
        threshold=0.5,
    )

    assert len(comp_df) == 2
    for col in ["pr_auc", "roc_auc", "precision", "recall", "f1_score", "false_positives"]:
        assert col in comp_df.columns
        assert not comp_df[col].isnull().any()
        assert (comp_df["pr_auc"] >= 0.70).all()
