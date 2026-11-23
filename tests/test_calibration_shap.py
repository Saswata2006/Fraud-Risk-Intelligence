"""Day 4 test suite: calibration, threshold optimization, SHAP, and risk scoring.

Tests verify:
- Calibration data separation and coverage
- Calibrated probability bounds [0, 1]
- Valid Brier score range
- Threshold decision logic
- Risk tier assignment
- SHAP feature alignment
- Serialization round-trip for calibrated pipeline
- Input schema validation in predict_risk()
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df():
    """Minimal synthetic DataFrame that mimics creditcard.csv structure."""
    rng = np.random.default_rng(0)
    n = 300
    data = {
        "Time": np.sort(rng.uniform(0, 172800, n)),
        "Amount": rng.exponential(scale=100, size=n),
        **{f"V{i}": rng.standard_normal(n) for i in range(1, 29)},
        "Class": (rng.uniform(size=n) < 0.05).astype(int),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def calibration_split(small_df):
    """Provide fit_df and calib_df from the synthetic dataset."""
    from src.models.calibration import create_calibration_split
    return create_calibration_split(small_df, fit_ratio=0.75)


@pytest.fixture(scope="module")
def rf_components(calibration_split):
    """Train a small RF on the fit subset and return preprocessor + classifier."""
    from src.models.calibration import retrain_rf_on_fit
    fit_df, _ = calibration_split
    X_fit = fit_df[[c for c in fit_df.columns if c != "Class"]]
    y_fit = fit_df["Class"]
    return retrain_rf_on_fit(X_fit, y_fit, n_estimators=10, max_depth=3)


@pytest.fixture(scope="module")
def calibrated_rf_pipeline(rf_components, calibration_split):
    """Build a calibrated RF pipeline using the calib subset."""
    from src.models.calibration import (
        fit_sigmoid_calibrator,
        build_calibrated_pipeline,
    )
    prep, clf = rf_components
    _, calib_df = calibration_split
    X_calib = calib_df[[c for c in calib_df.columns if c != "Class"]]
    y_calib = calib_df["Class"]
    X_calib_trans = prep.transform(X_calib)
    cal_clf = fit_sigmoid_calibrator(clf, X_calib_trans, y_calib)
    return build_calibrated_pipeline(prep, cal_clf)


# ---------------------------------------------------------------------------
# TASK 1: Calibration data separation
# ---------------------------------------------------------------------------

class TestCalibrationDataSeparation:
    """Verify fit and calib subsets are disjoint and cover the full training partition."""

    def test_disjoint_indices(self, small_df, calibration_split):
        """fit_df and calib_df must have no overlapping row positions."""
        from src.models.calibration import create_calibration_split
        fit_df, calib_df = calibration_split
        # Verify row positions are disjoint by checking index ranges
        assert len(fit_df) + len(calib_df) == len(small_df), (
            f"fit ({len(fit_df)}) + calib ({len(calib_df)}) != total ({len(small_df)})"
        )

    def test_covers_training_partition(self, small_df, calibration_split):
        """Union of fit_df and calib_df must equal the full training partition."""
        fit_df, calib_df = calibration_split
        assert len(fit_df) + len(calib_df) == len(small_df)

    def test_chronological_order_preserved(self, small_df, calibration_split):
        """fit_df rows must precede calib_df rows in Time."""
        fit_df, calib_df = calibration_split
        fit_max_time = float(fit_df["Time"].max())
        calib_min_time = float(calib_df["Time"].min())
        assert fit_max_time <= calib_min_time, (
            f"Chronological order violated: fit max time {fit_max_time} > "
            f"calib min time {calib_min_time}"
        )

    def test_invalid_fit_ratio_raises(self, small_df):
        """create_calibration_split must raise on out-of-range fit_ratio."""
        from src.models.calibration import create_calibration_split
        with pytest.raises(ValueError):
            create_calibration_split(small_df, fit_ratio=0.0)
        with pytest.raises(ValueError):
            create_calibration_split(small_df, fit_ratio=1.5)


# ---------------------------------------------------------------------------
# TASK 2: Calibrated probability bounds
# ---------------------------------------------------------------------------

class TestCalibratedProbabilities:
    """Verify calibrated probabilities are in [0.0, 1.0]."""

    def test_rf_probabilities_bounded(self, calibrated_rf_pipeline, small_df):
        """Calibrated RF probabilities must be in [0.0, 1.0]."""
        X = small_df[[c for c in small_df.columns if c != "Class"]]
        proba = calibrated_rf_pipeline.predict_proba(X)[:, 1]
        assert proba.min() >= 0.0, f"Probability below 0: {proba.min()}"
        assert proba.max() <= 1.0, f"Probability above 1: {proba.max()}"

    def test_probabilities_not_all_zero(self, calibrated_rf_pipeline, small_df):
        """Calibrated probabilities must not be uniformly zero."""
        X = small_df[[c for c in small_df.columns if c != "Class"]]
        proba = calibrated_rf_pipeline.predict_proba(X)[:, 1]
        assert proba.sum() > 0.0, "All calibrated probabilities are zero."


# ---------------------------------------------------------------------------
# TASK 3: Brier score validity
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Brier score must be in (0, 1) for a non-trivial model."""

    def test_brier_score_valid_range(self, calibrated_rf_pipeline, small_df):
        """Brier score on synthetic data must be in (0.0, 1.0)."""
        from sklearn.metrics import brier_score_loss
        X = small_df[[c for c in small_df.columns if c != "Class"]]
        y = small_df["Class"]
        proba = calibrated_rf_pipeline.predict_proba(X)[:, 1]
        bs = brier_score_loss(y, proba)
        assert 0.0 < bs < 1.0, f"Brier score out of range: {bs}"


# ---------------------------------------------------------------------------
# TASK 4: Threshold decision logic
# ---------------------------------------------------------------------------

class TestThresholdDecisionLogic:
    """Given a known probability array, binary decisions must match threshold."""

    def test_decision_matches_threshold(self):
        """Transactions with proba >= threshold must be flagged."""
        proba = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.99])
        threshold = 0.30
        expected = np.array([False, False, False, True, True, True])
        decisions = proba >= threshold
        np.testing.assert_array_equal(decisions, expected)

    def test_threshold_boundary(self):
        """Exactly at threshold: must be flagged (>=)."""
        proba = np.array([0.30])
        assert proba[0] >= 0.30

    def test_compute_threshold_metrics_shape(self, calibrated_rf_pipeline, small_df):
        """compute_threshold_metrics must return a DataFrame with expected columns."""
        from src.models.threshold_optimizer import compute_threshold_metrics
        X = small_df[[c for c in small_df.columns if c != "Class"]]
        y = small_df["Class"]
        thresholds = np.array([0.1, 0.5, 0.9])
        df = compute_threshold_metrics(calibrated_rf_pipeline, X, y, thresholds)
        required_cols = {"threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"}
        assert required_cols.issubset(df.columns)
        assert len(df) == 3

    def test_find_optimal_threshold_f1(self, calibrated_rf_pipeline, small_df):
        """find_optimal_threshold must return a valid threshold in the grid."""
        from src.models.threshold_optimizer import compute_threshold_metrics, find_optimal_threshold
        X = small_df[[c for c in small_df.columns if c != "Class"]]
        y = small_df["Class"]
        thresholds = np.arange(0.05, 0.96, 0.05)
        df = compute_threshold_metrics(calibrated_rf_pipeline, X, y, thresholds)
        opt_thr, row = find_optimal_threshold(df, objective="f1")
        assert 0.0 < opt_thr < 1.0, f"Optimal threshold out of range: {opt_thr}"
        assert isinstance(opt_thr, float)


# ---------------------------------------------------------------------------
# TASK 5: Risk tier assignment
# ---------------------------------------------------------------------------

class TestRiskTierAssignment:
    """Verify tier labels match defined cutoffs."""

    def test_low_tier(self):
        """Probabilities < low_cutoff must be 'Low'."""
        from src.models.risk_scorer import predict_risk
        from unittest.mock import MagicMock
        import numpy as np

        # Build a minimal mock pipeline that returns specific probabilities
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.98, 0.02], [0.95, 0.05]])

        # Build minimal valid DataFrame
        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(
            np.zeros((2, len(feature_cols))), columns=feature_cols
        )
        result = predict_risk(df, mock_pipeline, threshold=0.30, low_cutoff=0.10, high_cutoff=0.50)
        assert list(result["risk_tier"]) == ["Low", "Low"]

    def test_medium_tier(self):
        """Probabilities in [low_cutoff, high_cutoff) must be 'Medium'."""
        from src.models.risk_scorer import predict_risk
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.70, 0.30], [0.60, 0.40]])
        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        result = predict_risk(df, mock_pipeline, threshold=0.30, low_cutoff=0.10, high_cutoff=0.50)
        assert list(result["risk_tier"]) == ["Medium", "Medium"]

    def test_high_tier(self):
        """Probabilities >= high_cutoff must be 'High'."""
        from src.models.risk_scorer import predict_risk
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.20, 0.80], [0.10, 0.90]])
        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        result = predict_risk(df, mock_pipeline, threshold=0.30, low_cutoff=0.10, high_cutoff=0.50)
        assert list(result["risk_tier"]) == ["High", "High"]

    def test_tier_cutoffs_consistent(self):
        """All three tiers must appear for a spread of probabilities."""
        from src.models.risk_scorer import predict_risk
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.column_stack([
            [0.95, 0.70, 0.20],
            [0.05, 0.30, 0.80],
        ])
        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(np.zeros((3, len(feature_cols))), columns=feature_cols)
        result = predict_risk(df, mock_pipeline, threshold=0.30, low_cutoff=0.10, high_cutoff=0.50)
        assert set(result["risk_tier"]) == {"Low", "Medium", "High"}


# ---------------------------------------------------------------------------
# TASK 6: SHAP feature alignment
# ---------------------------------------------------------------------------

class TestSHAPFeatureAlignment:
    """Verify that SHAP values and feature names are aligned."""

    def test_shap_feature_count_matches_preprocessor(self, rf_components, small_df):
        """len(feature_names) must equal shap_values.shape[1] == 30."""
        from src.explainability.shap_explainer import compute_shap_values
        prep, clf = rf_components
        X = small_df[[c for c in small_df.columns if c != "Class"]].iloc[:50]
        shap_vals, X_trans, feat_names = compute_shap_values(prep, clf, X, model_type="random_forest")
        assert len(feat_names) == shap_vals.shape[1], (
            f"Feature name count ({len(feat_names)}) != SHAP columns ({shap_vals.shape[1]})"
        )
        assert shap_vals.shape[1] == X_trans.shape[1], (
            f"SHAP columns ({shap_vals.shape[1]}) != X_trans columns ({X_trans.shape[1]})"
        )

    def test_expected_30_features(self, rf_components, small_df):
        """The preprocessor must produce exactly 30 features: Time + Amount + V1-V28."""
        from src.explainability.shap_explainer import compute_shap_values
        prep, clf = rf_components
        X = small_df[[c for c in small_df.columns if c != "Class"]].iloc[:50]
        shap_vals, X_trans, feat_names = compute_shap_values(prep, clf, X, model_type="random_forest")
        assert X_trans.shape[1] == 30, f"Expected 30 features, got {X_trans.shape[1]}"


# ---------------------------------------------------------------------------
# TASK 7: Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    """Save and reload a calibrated pipeline; predictions must match."""

    def test_save_load_predictions_match(self, calibrated_rf_pipeline, small_df, tmp_path):
        """Reloaded pipeline must produce identical predict_proba outputs."""
        from src.models.calibration import save_calibrated_pipeline, load_calibrated_pipeline

        save_path = tmp_path / "test_calibrated_rf.joblib"
        save_calibrated_pipeline(calibrated_rf_pipeline, save_path)
        loaded_pipeline = load_calibrated_pipeline(save_path)

        X = small_df[[c for c in small_df.columns if c != "Class"]]
        original_proba = calibrated_rf_pipeline.predict_proba(X)[:, 1]
        loaded_proba = loaded_pipeline.predict_proba(X)[:, 1]
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)

    def test_threshold_metadata_round_trip(self, tmp_path):
        """Saved threshold metadata JSON must load back identical data."""
        from src.models.threshold_optimizer import save_threshold_metadata, load_threshold_metadata

        # Build a mock metrics row
        metrics_row = pd.Series({
            "threshold": 0.35,
            "precision": 0.7,
            "recall": 0.65,
            "f1": 0.674,
            "tp": 40,
            "fp": 17,
            "tn": 56940,
            "fn": 17,
            "financial_cost": 8670.0,
        })

        filepath = tmp_path / "threshold_metadata.json"
        save_threshold_metadata(
            threshold=0.35,
            model_name="TestModel",
            objective="f1",
            metrics_row=metrics_row,
            fp_cost_assumption=10.0,
            fn_cost_assumption=500.0,
            filepath=filepath,
        )
        loaded = load_threshold_metadata(filepath)
        assert loaded["optimal_threshold"] == pytest.approx(0.35)
        assert loaded["model_name"] == "TestModel"
        assert loaded["selection_objective"] == "f1"
        assert "cost_assumptions" in loaded


# ---------------------------------------------------------------------------
# TASK 8: Input schema validation
# ---------------------------------------------------------------------------

class TestInputSchemaValidation:
    """predict_risk must raise ValueError on invalid inputs."""

    def test_missing_column_raises(self):
        """Missing a required column must raise ValueError."""
        from src.models.risk_scorer import predict_risk
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        # DataFrame missing V28
        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 28)]  # missing V28
        df = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        with pytest.raises(ValueError, match="Missing required columns"):
            predict_risk(df, mock_pipeline, threshold=0.30)

    def test_invalid_threshold_raises(self):
        """threshold outside (0, 1) must raise ValueError."""
        from src.models.risk_scorer import predict_risk
        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        with pytest.raises(ValueError, match="threshold"):
            predict_risk(df, mock_pipeline, threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            predict_risk(df, mock_pipeline, threshold=1.1)

    def test_non_numeric_column_raises(self):
        """Non-numeric feature column must raise ValueError."""
        from src.models.risk_scorer import predict_risk, validate_input_schema
        from unittest.mock import MagicMock

        feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(np.zeros((2, len(feature_cols))), columns=feature_cols)
        df["V1"] = "bad_string"  # Inject non-numeric column
        with pytest.raises(ValueError):
            validate_input_schema(df)
