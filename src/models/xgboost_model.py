"""XGBoost modeling and temporal hyperparameter tuning module.

Constructs, tunes (using TimeSeriesSplit on training data only), trains,
and serializes an XGBClassifier pipeline calibrated with scale_pos_weight.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.features.preprocessor import create_feature_preprocessor


DEFAULT_XGB_ARTIFACT_PATH = Path("artifacts/xgboost_pipeline.joblib")


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Calculate scale_pos_weight from training class counts only.

    Args:
        y_train: Training target labels (0 and 1).

    Returns:
        Float ratio of legitimate to fraudulent transactions.
    """
    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)


def build_xgboost_pipeline(
    preprocessor: Optional[ColumnTransformer] = None,
    scale_pos_weight: Optional[float] = None,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    n_estimators: int = 150,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    tree_method: str = "hist",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a scikit-learn Pipeline with preprocessor and XGBClassifier.

    Args:
        preprocessor: Optional ColumnTransformer.
        scale_pos_weight: Class balancing weight for minority class.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        n_estimators: Number of boosting rounds.
        subsample: Row subsampling ratio.
        colsample_bytree: Column subsampling ratio per tree.
        tree_method: Tree construction algorithm (default 'hist').
        random_state: Seed for reproducibility.
        n_jobs: Parallel threads.

    Returns:
        Unfitted Pipeline.
    """
    if preprocessor is None:
        preprocessor = create_feature_preprocessor()

    classifier = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
        tree_method=tree_method,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    return pipeline


def tune_xgboost_chronological(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
    n_splits: int = 3,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Perform temporal cross-validation on training partition only.

    Respects chronological order using TimeSeriesSplit to evaluate candidates
    without forward-looking temporal leakage.

    Args:
        X_train: Training features.
        y_train: Training labels.
        preprocessor: Optional ColumnTransformer.
        n_splits: Number of chronological folds.

    Returns:
        Tuple of (best_params_dict, tuning_results_df).
    """
    if preprocessor is None:
        preprocessor = create_feature_preprocessor()

    # Pre-transform training data to avoid repeated preprocessing in CV loop
    X_train_trans = preprocessor.fit_transform(X_train)

    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 150, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 100, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 100, "subsample": 1.0, "colsample_bytree": 1.0},
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []

    for idx, params in enumerate(param_grid):
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_trans)):
            X_tr, y_tr = X_train_trans[train_idx], y_train.iloc[train_idx]
            X_val_f, y_val_f = X_train_trans[val_idx], y_train.iloc[val_idx]

            fold_spw = compute_scale_pos_weight(y_tr)

            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=fold_spw,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val_f)[:, 1]
            fold_ap = average_precision_score(y_val_f, probs)
            fold_scores.append(fold_ap)

        mean_ap = float(np.mean(fold_scores))
        std_ap = float(np.std(fold_scores))

        rec = dict(params)
        rec["mean_cv_pr_auc"] = round(mean_ap, 4)
        rec["std_cv_pr_auc"] = round(std_ap, 4)
        records.append(rec)

    results_df = pd.DataFrame(records).sort_values(by="mean_cv_pr_auc", ascending=False).reset_index(drop=True)
    best_params = results_df.iloc[0].to_dict()
    # Remove cv metrics from model kwargs
    best_params.pop("mean_cv_pr_auc", None)
    best_params.pop("std_cv_pr_auc", None)
    # Ensure proper int types for estimators and depth
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["n_estimators"] = int(best_params["n_estimators"])

    return best_params, results_df


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Optional[ColumnTransformer] = None,
    scale_pos_weight: Optional[float] = None,
    **params,
) -> Pipeline:
    """Train XGBoost pipeline on training partition.

    Args:
        X_train: Training features.
        y_train: Training labels.
        preprocessor: Optional ColumnTransformer.
        scale_pos_weight: Balancing weight. If None, calculated from y_train.
        **params: Hyperparameters for XGBClassifier.

    Returns:
        Fitted Pipeline.
    """
    if scale_pos_weight is None:
        scale_pos_weight = compute_scale_pos_weight(y_train)

    pipeline = build_xgboost_pipeline(
        preprocessor=preprocessor,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def get_xgb_feature_importances(
    pipeline: Pipeline,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract gain feature importances from fitted XGBoost pipeline.

    Args:
        pipeline: Fitted Pipeline.
        feature_names: Optional list of feature names.

    Returns:
        pd.DataFrame sorted descending by importance.
    """
    xgb_clf: xgb.XGBClassifier = pipeline.named_steps["classifier"]

    if feature_names is None:
        try:
            feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(xgb_clf.feature_importances_))]

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": xgb_clf.feature_importances_,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp


def save_xgb_model(
    pipeline: Pipeline,
    filepath: Union[str, Path] = DEFAULT_XGB_ARTIFACT_PATH,
) -> Path:
    """Serialize and save XGBoost pipeline.

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


def load_xgb_model(
    filepath: Union[str, Path] = DEFAULT_XGB_ARTIFACT_PATH,
) -> Pipeline:
    """Load serialized XGBoost pipeline.

    Args:
        filepath: Path to artifact.

    Returns:
        Fitted Pipeline.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path.resolve()}")
    return joblib.load(path)
