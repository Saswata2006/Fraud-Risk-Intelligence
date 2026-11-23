"""Day 3 Pipeline Execution Script: Tree Ensembles (Random Forest & XGBoost), Benchmarking & Model Selection.

Executes:
1. Loading chronological train/validation/test partitions.
2. Loading baseline Logistic Regression artifact.
3. Training Random Forest with class_weight='balanced_subsample'.
4. Chronological hyperparameter tuning and training of XGBoost with scale_pos_weight.
5. Unified benchmarking across all 3 models on the validation partition.
6. Generating combined Precision-Recall curve and Feature Importance plots.
7. Model selection and saving all artifacts.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.data_loader import load_raw_data
from src.features.data_split import create_chronological_splits
from src.models.baseline import load_model as load_baseline_model
from src.models.random_forest import (
    train_random_forest_model,
    save_rf_model,
)
from src.models.xgboost_model import (
    tune_xgboost_chronological,
    train_xgboost_model,
    save_xgb_model,
    compute_scale_pos_weight,
)
from src.models.model_comparison import (
    benchmark_models,
    plot_multi_model_pr_curves,
    plot_feature_importances_comparison,
    save_comparison_metrics,
)


def main():
    print("=" * 68)
    print("DAY 3: RANDOM FOREST, XGBOOST, BENCHMARKING & MODEL SELECTION")
    print("=" * 68)

    # 1. Ingestion & Split
    print("\n[1/6] Loading dataset and creating chronological partitions...")
    df = load_raw_data(ROOT_DIR / "data" / "raw" / "creditcard.csv")
    train_df, val_df, test_df = create_chronological_splits(df)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]

    print(f"Train Partition : {len(train_df):,} samples | Fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
    print(f"Val Partition   : {len(val_df):,} samples | Fraud: {y_val.sum():,} ({y_val.mean()*100:.4f}%)")
    print(f"Test Partition  : {len(test_df):,} samples | (Held out, untouched)")

    # 2. Load Day 2 Baseline
    print("\n[2/6] Loading existing baseline Logistic Regression pipeline...")
    baseline_pipe = load_baseline_model(ROOT_DIR / "artifacts" / "baseline_logistic_regression.joblib")
    print("Loaded baseline Logistic Regression successfully.")

    # 3. Train Random Forest
    print("\n[3/6] Training Random Forest (class_weight='balanced_subsample', max_depth=10, n_est=100)...")
    rf_pipe = train_random_forest_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf_artifact_path = save_rf_model(rf_pipe, ROOT_DIR / "artifacts" / "random_forest_pipeline.joblib")
    print(f"Saved Random Forest Pipeline: {rf_artifact_path.relative_to(ROOT_DIR)}")

    # 4. Tune & Train XGBoost
    print("\n[4/6] Tuning and training XGBoost with scale_pos_weight on training folds...")
    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    print("Running TimeSeriesSplit(n_splits=3) chronological cross-validation...")
    best_params, cv_results = tune_xgboost_chronological(X_train, y_train, n_splits=3)
    print("Hyperparameter tuning results on training folds:")
    print(cv_results.to_string(index=False))
    print(f"\nSelected Best Parameters: {best_params}")

    xgb_pipe = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        **best_params,
    )
    xgb_artifact_path = save_xgb_model(xgb_pipe, ROOT_DIR / "artifacts" / "xgboost_pipeline.joblib")
    print(f"Saved XGBoost Pipeline: {xgb_artifact_path.relative_to(ROOT_DIR)}")

    # 5. Benchmark Models
    print("\n[5/6] Benchmarking all 3 models on the SAME validation partition (threshold = 0.5)...")
    model_dict = {
        "Logistic Regression (Baseline)": baseline_pipe,
        "Random Forest": rf_pipe,
        "XGBoost": xgb_pipe,
    }

    comparison_df = benchmark_models(model_dict, X_val, y_val, threshold=0.5)
    print("\n" + "=" * 95)
    print("VALIDATION BENCHMARK RESULTS (Threshold = 0.5)")
    print("=" * 95)
    display_cols = ["model", "pr_auc", "roc_auc", "precision", "recall", "f1_score", "true_positives", "false_positives", "false_negatives"]
    print(comparison_df[display_cols].to_string(index=False))

    j_path, c_path = save_comparison_metrics(
        comparison_df,
        json_path=ROOT_DIR / "artifacts" / "model_comparison_metrics.json",
        csv_path=ROOT_DIR / "artifacts" / "model_comparison_metrics.csv",
    )
    print(f"\nSaved Comparison Metrics: {j_path.relative_to(ROOT_DIR)} & {c_path.relative_to(ROOT_DIR)}")

    # 6. Generate Comparison Visualizations
    print("\n[6/6] Generating multi-model comparison charts...")
    fig_dir = ROOT_DIR / "reports" / "figures"
    pr_png, pr_html = plot_multi_model_pr_curves(model_dict, X_val, y_val, fig_dir)
    imp_png, imp_html = plot_feature_importances_comparison(rf_pipe, xgb_pipe, fig_dir, top_n=12)

    print(f"Generated Visualizations in reports/figures/:")
    print(f"  - {Path(pr_png).name} & {Path(pr_html).name}")
    print(f"  - {Path(imp_png).name} & {Path(imp_html).name}")

    # Best Model Selection Summary
    best_model_row = comparison_df.iloc[0]
    print("\n" + "=" * 68)
    print("MODEL SELECTION SUMMARY")
    print("=" * 68)
    print(f"Best Candidate by PR-AUC: {best_model_row['model']} (PR-AUC = {best_model_row['pr_auc']:.4f})")
    print(f"  - PR-AUC: {best_model_row['pr_auc']:.4f}")
    print(f"  - ROC-AUC: {best_model_row['roc_auc']:.4f}")
    print(f"  - Fraud Caught (Recall): {best_model_row['recall']*100:.2f}% ({best_model_row['true_positives']}/57 cases)")
    print(f"  - False Positives: {best_model_row['false_positives']:,}")

    print("\n" + "=" * 68)
    print("DAY 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 68)


if __name__ == "__main__":
    main()
