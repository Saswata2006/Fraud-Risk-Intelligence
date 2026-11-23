"""Day 2 Pipeline Execution Script: Leakage-Safe Splitting, Feature Engineering & Baseline Logistic Regression.

Executes:
1. Chronological data splitting (60% train, 20% validation, 20% test).
2. Cross-partition duplicate leakage audit.
3. Feature engineering pipeline fitted strictly on training data.
4. Baseline Logistic Regression training with class_weight='balanced'.
5. Rigorous validation evaluation (PR-AUC, ROC-AUC, Recall, Precision, F1, Confusion Matrix).
6. Diagnostics plot generation and artifact serialization.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.data_loader import load_raw_data
from src.features.data_split import (
    create_chronological_splits,
    audit_cross_partition_duplicates,
    filter_cross_partition_duplicates,
)
from src.features.preprocessor import create_feature_preprocessor
from src.models.baseline import train_baseline_model, save_model
from src.models.evaluation import (
    evaluate_model,
    plot_precision_recall_curve,
    plot_confusion_matrix_heatmap,
    plot_prediction_score_distribution,
    save_metrics_to_json,
)


def main():
    print("=" * 65)
    print("DAY 2: DATA SPLITTING, FEATURE ENGINEERING & BASELINE MODEL")
    print("=" * 65)

    # 1. Ingestion
    print("\n[1/5] Loading raw data from data/raw/creditcard.csv...")
    df = load_raw_data(ROOT_DIR / "data" / "raw" / "creditcard.csv")
    print(f"Loaded DataFrame with shape: {df.shape}")

    # 2. Chronological Splitting
    print("\n[2/5] Sorting chronologically by Time and partitioning (60/20/20)...")
    train_df, val_df, test_df = create_chronological_splits(df, 0.60, 0.20, 0.20)
    
    # Audit cross-partition leakage
    split_report = audit_cross_partition_duplicates(train_df, val_df, test_df)
    
    print("\n--- Partition Class Distribution ---")
    print(f"Train Partition : {split_report.train_count:,} samples | Fraud: {split_report.train_fraud_count:,} ({split_report.train_fraud_rate:.4f}%)")
    print(f"Val Partition   : {split_report.val_count:,} samples | Fraud: {split_report.val_fraud_count:,} ({split_report.val_fraud_rate:.4f}%)")
    print(f"Test Partition  : {split_report.test_count:,} samples | Fraud: {split_report.test_fraud_count:,} ({split_report.test_fraud_rate:.4f}%)")

    print("\n--- Cross-Partition Duplicate Leakage Audit ---")
    print(f"Full-Row Duplicate Overlap (Train <-> Val)  : {split_report.full_row_cross_train_val}")
    print(f"Full-Row Duplicate Overlap (Train <-> Test) : {split_report.full_row_cross_train_test}")
    print(f"Full-Row Duplicate Overlap (Val <-> Test)   : {split_report.full_row_cross_val_test}")
    print(f"Feature Profiles (V1-V28, Amount) Train <-> Val Overlap : {split_report.feature_profile_cross_train_val} profiles")
    print(f"  -> Fraud cases among shared feature profiles in Val  : {split_report.val_leaked_fraud_count} (all are legitimate transactions)")

    # 3. Feature Engineering & Preprocessing
    print("\n[3/5] Setting up leakage-safe ColumnTransformer (log1p Amount + RobustScaler)...")
    preprocessor = create_feature_preprocessor(use_log1p_amount=True, scaler_type="robust")

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]

    X_val = val_df.drop(columns=["Class"])
    y_val = val_df["Class"]

    # 4. Model Training
    print("\n[4/5] Training Logistic Regression (class_weight='balanced', random_state=42)...")
    pipeline = train_baseline_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        C=1.0,
        max_iter=1000,
        random_state=42,
    )

    # 5. Model Evaluation (Validation Split ONLY)
    print("\n[5/5] Evaluating performance on Validation split ONLY (Test remains untouched)...")
    metrics = evaluate_model(pipeline, X_val, y_val, threshold=0.5)

    print("\n=== BASELINE VALIDATION METRICS (Threshold = 0.5) ===")
    print(f"Average Precision (PR-AUC) : {metrics['average_precision_pr_auc']:.4f}")
    print(f"ROC-AUC                    : {metrics['roc_auc']:.4f}")
    print(f"Precision                  : {metrics['precision']:.4f}")
    print(f"Recall                     : {metrics['recall']:.4f} ({metrics['confusion_matrix']['true_positives']}/{metrics['actual_fraud_samples']} fraud detected)")
    print(f"F1-Score                   : {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(f"  TN: {metrics['confusion_matrix']['true_negatives']:,}  |  FP: {metrics['confusion_matrix']['false_positives']:,}")
    print(f"  FN: {metrics['confusion_matrix']['false_negatives']:,}        |  TP: {metrics['confusion_matrix']['true_positives']:,}")

    # Comparative evaluation without shared feature profiles
    clean_val, leaked_val = filter_cross_partition_duplicates(val_df, train_df)
    clean_metrics = evaluate_model(pipeline, clean_val.drop(columns=["Class"]), clean_val["Class"], threshold=0.5)
    print(f"\nAudit: PR-AUC on Deduplicated Validation Subset ({len(clean_val):,} samples): {clean_metrics['average_precision_pr_auc']:.4f}")
    print(f"       Recall on Deduplicated Validation Subset: {clean_metrics['recall']:.4f}")

    # Serialize Artifacts
    print("\n--- Serializing Artifacts & Visualizations ---")
    model_path = save_model(pipeline, ROOT_DIR / "artifacts" / "baseline_logistic_regression.joblib")
    metrics_path = save_metrics_to_json(metrics, ROOT_DIR / "artifacts" / "baseline_metrics.json")
    print(f"Saved Model Pipeline  : {model_path.relative_to(ROOT_DIR)}")
    print(f"Saved Metrics JSON    : {metrics_path.relative_to(ROOT_DIR)}")

    val_probs = pipeline.predict_proba(X_val)[:, 1]
    fig_dir = ROOT_DIR / "reports" / "figures"
    p8_mpl, p8_html = plot_precision_recall_curve(y_val, val_probs, fig_dir)
    p9_mpl, p9_html = plot_confusion_matrix_heatmap(metrics["confusion_matrix"], threshold=0.5, output_dir=fig_dir)
    p10_mpl, p10_html = plot_prediction_score_distribution(y_val, val_probs, threshold=0.5, output_dir=fig_dir)

    print(f"Generated Visualizations in reports/figures/:")
    print(f"  - {Path(p8_mpl).name} & {Path(p8_html).name}")
    print(f"  - {Path(p9_mpl).name} & {Path(p9_html).name}")
    print(f"  - {Path(p10_mpl).name} & {Path(p10_html).name}")

    print("\n" + "=" * 65)
    print("DAY 2 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
