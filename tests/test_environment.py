"""Environment and dependency verification tests.

Validates that all required core libraries are installed, compatible, and
functional in the Python 3.12 environment.
"""

import sys
import numpy as np
import pandas as pd
import duckdb
import sklearn
import xgboost as xgb
import shap
import plotly
import streamlit
import pytest


def test_python_version():
    """Verify runtime is Python 3.12+."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 12, f"Expected Python >= 3.12, got {sys.version}"


def test_numpy_pandas_integration():
    """Verify numpy and pandas can perform basic array and dataframe operations."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    df = pd.DataFrame({"amount": arr, "label": [0, 0, 1, 0, 0]})
    assert df["amount"].mean() == 3.0
    assert len(df) == 5
    assert df["label"].sum() == 1


def test_duckdb_in_memory():
    """Verify DuckDB executes in-memory queries properly."""
    con = duckdb.connect(database=":memory:")
    result = con.execute("SELECT 42 AS answer").fetchall()
    assert result[0][0] == 42
    con.close()


def test_scikit_learn_smoke():
    """Verify scikit-learn basic estimators work."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    clf = LogisticRegression()
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 50


def test_xgboost_smoke():
    """Verify XGBoost can instantiate and fit a simple classifier."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    model = xgb.XGBClassifier(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 4


def test_shap_import():
    """Verify SHAP module is functional."""
    assert hasattr(shap, "TreeExplainer")
    assert hasattr(shap, "summary_plot")


def test_plotly_smoke():
    """Verify Plotly figure construction."""
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
    assert fig is not None
    assert len(fig.data) == 1


def test_streamlit_smoke():
    """Verify Streamlit module imports cleanly."""
    assert hasattr(streamlit, "title")
    assert hasattr(streamlit, "sidebar")
