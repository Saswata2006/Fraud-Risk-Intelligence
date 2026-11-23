"""Unit tests to verify exploratory analysis charts generation."""

from pathlib import Path
import pytest

REPORTS_DIR = Path("reports/figures")

EXPECTED_FIGURES = [
    "01_class_distribution.png",
    "01_class_distribution.html",
    "02_amount_distribution_by_class.png",
    "02_amount_distribution_by_class.html",
    "03_transaction_frequency_time.png",
    "03_transaction_frequency_time.html",
    "04_fraud_rate_by_amount_bands.png",
    "04_fraud_rate_by_amount_bands.html",
    "05_top_feature_correlations.png",
    "05_top_feature_correlations.html",
    "06_key_anonymized_features_distribution.png",
    "06_key_anonymized_features_distribution.html",
    "07_correlation_heatmap.png",
]


def test_generated_figures_exist():
    """Ensure all required static and interactive figures exist and have non-zero size."""
    for fig_name in EXPECTED_FIGURES:
        fig_path = REPORTS_DIR / fig_name
        assert fig_path.exists(), f"Figure file {fig_name} does not exist"
        assert fig_path.stat().st_size > 0, f"Figure file {fig_name} is empty"
