"""Random Forest modeling module for Credit Card Fraud Detection.

Constructs, trains, and serializes a balanced Random Forest pipeline using
class_weight='balanced_subsample' to prevent majority class dominance while
maintaining tree diversity.
"""

from pathlib import Path
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.features.preprocessor import create_feature_preprocessor


DEFAULT_RF_ARTIFACT_PATH = Path("artifacts/random_forest_pipeline.joblib")


def build_random_forest_pipeline(
    preprocessor: Optional[ColumnTransformer] = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 2,
    class_weight: str = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessor and RandomForestClassifier.

    Args:
        preprocessor: Optional pre-configured ColumnTransformer.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples required at each leaf node.
        class_weight: Strategy for weighting minority class across bootstrap samples.
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs.

    Returns:
        Unfitted Pipeline.
    """
    if preprocessor is None:
        preprocessor = create_feature_preprocessor()

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


def train_random_forest_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Train Random Forest pipeline strictly on training split.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        preprocessor: Optional ColumnTransformer.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples per leaf.
        random_state: Random state seed.
        n_jobs: Worker threads.

    Returns:
        Fitted Pipeline.
    """
    pipeline = build_random_forest_pipeline(
        preprocessor=preprocessor,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def get_rf_feature_importances(
    pipeline: Pipeline,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract Gini feature importances from fitted Random Forest pipeline.

    Args:
        pipeline: Fitted Pipeline containing 'preprocessor' and 'classifier'.
        feature_names: Optional list of feature names. If None, derived from preprocessor.

    Returns:
        pd.DataFrame with 'feature' and 'importance' columns sorted descending.
    """
    rf_clf: RandomForestClassifier = pipeline.named_steps["classifier"]
    
    if feature_names is None:
        try:
            feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(rf_clf.feature_importances_))]

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_clf.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp


def save_rf_model(
    pipeline: Pipeline,
    filepath: Union[str, Path] = DEFAULT_RF_ARTIFACT_PATH,
) -> Path:
    """Serialize and save Random Forest pipeline.

    Args:
        pipeline: Fitted Pipeline.
        filepath: Target file path.

    Returns:
        Path to saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_rf_model(
    filepath: Union[str, Path] = DEFAULT_RF_ARTIFACT_PATH,
) -> Pipeline:
    """Load serialized Random Forest pipeline.

    Args:
        filepath: Path to artifact.

    Returns:
        Fitted Pipeline.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path.resolve()}")
    return joblib.load(path)
