"""Unit tests for baseline Logistic Regression modeling, preprocessing, and evaluation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.dataset
from sklearn.pipeline import Pipeline

from src.ingestion.data_loader import load_raw_data
from src.features.data_split import create_chronological_splits
from src.features.preprocessor import create_feature_preprocessor
from src.models.baseline import build_baseline_pipeline, train_baseline_model, load_model
from src.models.evaluation import evaluate_model

DATA_PATH = Path("data/raw/creditcard.csv")
MODEL_PATH = Path("artifacts/baseline_logistic_regression.joblib")


@pytest.fixture(scope="session")
def split_data():
    """Fixture providing chronological splits for modeling tests."""
    df = load_raw_data(DATA_PATH)
    train_df, val_df, test_df = create_chronological_splits(df, 0.60, 0.20, 0.20)
    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]
    return X_train, y_train, X_val, y_val


def test_feature_preprocessor_transformation(split_data):
    """Verify preprocessor transforms features properly without target leakage."""
    X_train, y_train, X_val, _ = split_data
    preprocessor = create_feature_preprocessor(use_log1p_amount=True, scaler_type="robust")

    # Verify fit only on X_train
    preprocessor.fit(X_train)
    transformed_train = preprocessor.transform(X_train)
    transformed_val = preprocessor.transform(X_val)

    assert transformed_train.shape == (len(X_train), 30)
    assert transformed_val.shape == (len(X_val), 30)
    assert not np.isnan(transformed_train).any()
    assert not np.isnan(transformed_val).any()


def test_model_prediction_shapes_and_probabilities(split_data):
    """Verify prediction shapes and valid probability bounds."""
    _, _, X_val, y_val = split_data
    pipeline = load_model(MODEL_PATH)

    preds = pipeline.predict(X_val)
    probs = pipeline.predict_proba(X_val)

    # 1. Output shapes
    assert preds.shape == (len(X_val),)
    assert probs.shape == (len(X_val), 2)

    # 2. Probability bounds [0.0, 1.0] and row sum = 1.0
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_model_reproducibility(split_data):
    """Verify identical models with random_state=42 yield exact same coefficients and predictions."""
    X_train, y_train, X_val, _ = split_data
    sample_X_train = X_train.iloc[:1000]
    sample_y_train = y_train.iloc[:1000]
    sample_X_val = X_val.iloc[:200]

    pipe1 = train_baseline_model(sample_X_train, sample_y_train, random_state=42)
    pipe2 = train_baseline_model(sample_X_train, sample_y_train, random_state=42)

    coef1 = pipe1.named_steps["classifier"].coef_
    coef2 = pipe2.named_steps["classifier"].coef_
    assert np.array_equal(coef1, coef2)

    prob1 = pipe1.predict_proba(sample_X_val)
    prob2 = pipe2.predict_proba(sample_X_val)
    assert np.array_equal(prob1, prob2)


def test_validation_metrics_performance(split_data):
    """Verify baseline model achieves expected performance thresholds on validation data."""
    _, _, X_val, y_val = split_data
    pipeline = load_model(MODEL_PATH)
    metrics = evaluate_model(pipeline, X_val, y_val, threshold=0.5)

    assert metrics["average_precision_pr_auc"] >= 0.70, f"Expected PR-AUC >= 0.70, got {metrics['average_precision_pr_auc']}"
    assert metrics["roc_auc"] >= 0.90, f"Expected ROC-AUC >= 0.90, got {metrics['roc_auc']}"
    assert metrics["recall"] >= 0.80, f"Expected Recall >= 0.80, got {metrics['recall']}"
    assert metrics["confusion_matrix"]["true_positives"] == 50
    assert metrics["confusion_matrix"]["false_negatives"] == 7
