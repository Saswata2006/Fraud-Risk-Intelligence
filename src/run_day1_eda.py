"""Driver script for Day 1: Data Ingestion, Validation, SQL Analytics & Visualizations.

Executes the complete Day 1 pipeline reproducibly.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingestion.data_loader import load_raw_data, validate_dataset
from src.ingestion.sql_analytics import run_all_sql_analytics
from src.features.eda_analysis import generate_all_visualizations


def main():
    print("=" * 60)
    print("DAY 1: CREDIT CARD FRAUD DATA INGESTION & EDA PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading raw data from data/raw/creditcard.csv...")
    df = load_raw_data("data/raw/creditcard.csv")
    print(f"Loaded DataFrame with shape: {df.shape}")

    # 2. Validation
    print("\n[2/4] Validating schema, missing values, duplicates, and ranges...")
    report = validate_dataset(df)
    print(f"Dataset Valid: {report.is_valid}")
    print(f"Total Rows: {report.total_rows:,} | Total Columns: {report.total_cols}")
    print(f"Missing Values: {report.missing_values_count}")
    print(f"Duplicate Rows (excluding 1st occurrence): {report.duplicate_rows_count:,}")
    print(f"  -> Duplicate Fraud Transactions: {report.duplicate_fraud_count}")
    print(f"  -> Duplicate Legit Transactions: {report.duplicate_legit_count:,}")
    print(f"Transactions with $0.00 Amount: {report.zero_amount_count:,}")
    print(f"  -> Zero Amount Fraud Transactions: {report.zero_amount_fraud_count}")
    print(f"Negative Amounts: {report.negative_amount_count}")
    print(f"Class Distribution: Legit={report.legit_count:,} ({(report.legit_count/report.total_rows)*100:.3f}%), Fraud={report.fraud_count} ({report.fraud_percentage:.3f}%)")

    # 3. DuckDB SQL Analytics
    print("\n[3/4] Running DuckDB in-memory analytical SQL queries...")
    sql_results = run_all_sql_analytics(df)
    
    print("\n--- DuckDB: Class Distribution & Volume ---")
    print(sql_results["class_counts"].to_string(index=False))

    print("\n--- DuckDB: Amount Statistics by Class ---")
    print(sql_results["amount_stats"].to_string(index=False))

    print("\n--- DuckDB: Fraud Rate by Amount Tier ---")
    print(sql_results["amount_tiers"].to_string(index=False))

    print("\n--- DuckDB: Hourly Fraud Frequency (Sample: First 6 Hours) ---")
    print(sql_results["hourly_frequency"].head(6).to_string(index=False))

    # 4. Generate Visualizations
    print("\n[4/4] Generating publication-quality charts (PNG & interactive HTML)...")
    figures = generate_all_visualizations(df, output_dir=ROOT_DIR / "reports" / "figures")
    print(f"Successfully generated {len(figures)} visualization files in reports/figures/:")
    for fig_path in figures:
        print(f"  - {Path(fig_path).name}")

    print("\n" + "=" * 60)
    print("DAY 1 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
