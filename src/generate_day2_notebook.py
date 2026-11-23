"""Generate Jupyter Notebook for Day 2 Baseline Modeling."""

import json
from pathlib import Path

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Day 2: Leakage-Safe Feature Engineering & Baseline Logistic Regression\n",
                "\n",
                "**Project**: Explainable Credit Card Fraud Detection & Risk Scoring  \n",
                "**Goal**: Chronological data splitting (60/20/20), cross-partition duplicate leakage auditing, scikit-learn Pipeline feature engineering, and baseline Logistic Regression modeling."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup & Environment Initialization\n",
                "Import core libraries and modular components authored under `src/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "ROOT_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                "if str(ROOT_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(ROOT_DIR))\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import plotly.express as px\n",
                "\n",
                "from src.ingestion.data_loader import load_raw_data\n",
                "from src.features.data_split import create_chronological_splits, audit_cross_partition_duplicates\n",
                "from src.features.preprocessor import create_feature_preprocessor\n",
                "from src.models.baseline import train_baseline_model, load_model, save_model\n",
                "from src.models.evaluation import evaluate_model\n",
                "\n",
                "pd.set_option('display.float_format', lambda x: '%.4f' % x)\n",
                "print('Environment initialized successfully.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Chronological Data Splitting & Leakage Audit\n",
                "We sort transactions chronologically by `Time` and partition into **60% Train**, **20% Validation**, and **20% Test**.\n",
                "Crucially, we audit cross-partition duplicate records to verify that test and validation partitions remain leak-free."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "DATA_PATH = ROOT_DIR / 'data' / 'raw' / 'creditcard.csv'\n",
                "df = load_raw_data(DATA_PATH)\n",
                "\n",
                "train_df, val_df, test_df = create_chronological_splits(df, 0.60, 0.20, 0.20)\n",
                "split_report = audit_cross_partition_duplicates(train_df, val_df, test_df)\n",
                "\n",
                "print(f\"Train : {split_report.train_count:,} rows | Fraud: {split_report.train_fraud_count:,} ({split_report.train_fraud_rate:.4f}%)\")\n",
                "print(f\"Val   : {split_report.val_count:,} rows | Fraud: {split_report.val_fraud_count:,} ({split_report.val_fraud_rate:.4f}%)\")\n",
                "print(f\"Test  : {split_report.test_count:,} rows | Fraud: {split_report.test_fraud_count:,} ({split_report.test_fraud_rate:.4f}%)\")\n",
                "print(f\"Full-Row Overlap (Train <-> Val): {split_report.full_row_cross_train_val}\")\n",
                "print(f\"Full-Row Overlap (Train <-> Test): {split_report.full_row_cross_train_test}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Feature Engineering & Preprocessing Pipeline\n",
                "We encapsulate all transformations inside a `ColumnTransformer`:\n",
                "- **Amount**: `log1p` transformation followed by `RobustScaler` to tame extreme skewness.\n",
                "- **Time**: `RobustScaler` to standardize scale without outlier distortion.\n",
                "- **V1–V28**: Passed through as standardized PCA features.\n",
                "- **Zero Leakage**: Fitted strictly on `X_train`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train = train_df.drop(columns=['Class'])\n",
                "y_train = train_df['Class']\n",
                "\n",
                "X_val = val_df.drop(columns=['Class'])\n",
                "y_val = val_df['Class']\n",
                "\n",
                "preprocessor = create_feature_preprocessor(use_log1p_amount=True, scaler_type='robust')\n",
                "print('Preprocessor definition:')\n",
                "print(preprocessor)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Baseline Logistic Regression Training\n",
                "We train Logistic Regression with `class_weight='balanced'` and `random_state=42`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pipeline = train_baseline_model(X_train, y_train, preprocessor=preprocessor, random_state=42)\n",
                "print('Fitted Pipeline:')\n",
                "print(pipeline)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Model Evaluation (Validation Set Only)\n",
                "We evaluate the model on unseen validation data using the initial threshold benchmark ($0.5$)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "metrics = evaluate_model(pipeline, X_val, y_val, threshold=0.5)\n",
                "print('=== Validation Evaluation Metrics ===')\n",
                "for k, v in metrics.items():\n",
                "    print(f\"{k}: {v}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Diagnostic Visualizations\n",
                "Visualizing Precision-Recall Curve, Confusion Matrix, and prediction score distributions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.models.evaluation import plot_precision_recall_curve, plot_confusion_matrix_heatmap, plot_prediction_score_distribution\n",
                "\n",
                "probs = pipeline.predict_proba(X_val)[:, 1]\n",
                "fig_dir = ROOT_DIR / 'reports' / 'figures'\n",
                "plot_precision_recall_curve(y_val, probs, fig_dir)\n",
                "plot_confusion_matrix_heatmap(metrics['confusion_matrix'], threshold=0.5, output_dir=fig_dir)\n",
                "plot_prediction_score_distribution(y_val, probs, threshold=0.5, output_dir=fig_dir)\n",
                "print('Generated all validation diagnostic plots.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Conclusions & Day 3 Roadmap\n",
                "- Baseline Logistic Regression establishes a strong ranking baseline (**PR-AUC = 0.7552**, **ROC-AUC = 0.9691**).\n",
                "- High recall (**87.72%**) catches 50/57 fraud cases, but default threshold yields 1,916 false positives due to balanced weighting.\n",
                "- Next step: Day 3/4 feature engineering and threshold optimization to minimize expected financial loss and elevate Precision."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.16"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path("notebooks/02_baseline_modeling.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Generated notebook at {output_path.resolve()}")
