"""Baseline Logistic Regression training module.

Constructs, trains, and serializes a reproducible baseline Logistic Regression
pipeline with class_weight='balanced' to handle severe class imbalance.
"""

from pathlib import Path
from typing import Optional, Union
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessor import create_feature_preprocessor


DEFAULT_ARTIFACT_PATH = Path("artifacts/baseline_logistic_regression.joblib")


def build_baseline_pipeline(
    preprocessor: Optional[ColumnTransformer] = None,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    solver: str = "lbfgs",
) -> Pipeline:
    """Construct an end-to-end Pipeline with preprocessing and Logistic Regression.

    Args:
        preprocessor: Optional pre-configured ColumnTransformer. If None, creates default.
        C: Inverse regularization strength parameter.
        max_iter: Maximum number of iterations for the solver to converge.
        random_state: Random state seed for reproducibility.
        solver: Optimization algorithm (default 'lbfgs').

    Returns:
        Unfitted sklearn Pipeline.
    """
    if preprocessor is None:
        preprocessor = create_feature_preprocessor()

    classifier = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver=solver,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Train the baseline Logistic Regression pipeline on the training split only.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        preprocessor: Optional ColumnTransformer.
        C: Regularization parameter.
        max_iter: Maximum iterations.
        random_state: Random state seed.

    Returns:
        Fitted sklearn Pipeline.
    """
    pipeline = build_baseline_pipeline(
        preprocessor=preprocessor,
        C=C,
        max_iter=max_iter,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(
    pipeline: Pipeline,
    filepath: Union[str, Path] = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Serialize and save the trained model pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        filepath: Destination file path.

    Returns:
        Path to saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(
    filepath: Union[str, Path] = DEFAULT_ARTIFACT_PATH,
) -> Pipeline:
    """Load a serialized model pipeline.

    Args:
        filepath: Path to saved artifact.

    Returns:
        Loaded Pipeline.

    Raises:
        FileNotFoundError: If artifact does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path.resolve()}")
    pipeline = joblib.load(path)
    return pipeline
