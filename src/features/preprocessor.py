"""Feature engineering and preprocessing pipeline module.

Constructs leakage-safe scikit-learn ColumnTransformers and Pipelines.
Transforms skewed transaction amounts (via log1p and robust scaling),
normalizes elapsed time, and preserves anonymized PCA feature components.
"""

from typing import List, Optional
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler


DEFAULT_PCA_FEATURES = [f"V{i}" for i in range(1, 29)]


def create_feature_preprocessor(
    pca_features: Optional[List[str]] = None,
    time_col: str = "Time",
    amount_col: str = "Amount",
    use_log1p_amount: bool = True,
    scale_pca: bool = False,
    scaler_type: str = "robust",
) -> ColumnTransformer:
    """Build a leakage-safe ColumnTransformer for credit card transactions.

    Preprocesses:
    - Time: Robust scaling (or standard scaling) to normalize scale without outlier sensitivity.
    - Amount: log1p transform (to stabilize high positive skewness) followed by scaling.
    - V1-V28: Passthrough (since PCA components are already standardized orthogonal projections),
      with optional standard scaling.

    Args:
        pca_features: List of column names for PCA features V1-V28.
        time_col: Column name for elapsed time.
        amount_col: Column name for transaction amount.
        use_log1p_amount: Whether to apply log(1 + x) transformation to Amount.
        scale_pca: Whether to apply StandardScaler to PCA features.
        scaler_type: Scaling algorithm ('robust' for RobustScaler, 'standard' for StandardScaler).

    Returns:
        ColumnTransformer instance ready to be integrated into an sklearn Pipeline.
    """
    if pca_features is None:
        pca_features = DEFAULT_PCA_FEATURES

    # Select base scaler class
    scaler_cls = RobustScaler if scaler_type.lower() == "robust" else StandardScaler

    # Time pipeline
    time_pipe = Pipeline([
        ("time_scaler", scaler_cls())
    ])

    # Amount pipeline
    amount_steps = []
    if use_log1p_amount:
        amount_steps.append(("log1p", FunctionTransformer(np.log1p, validate=True)))
    amount_steps.append(("amount_scaler", scaler_cls()))
    amount_pipe = Pipeline(amount_steps)

    # PCA features treatment
    pca_transformer = StandardScaler() if scale_pca else "passthrough"

    transformers = [
        ("time", time_pipe, [time_col]),
        ("amount", amount_pipe, [amount_col]),
        ("pca", pca_transformer, pca_features),
    ]

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
