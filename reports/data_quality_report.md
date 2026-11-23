# Data Quality & Profiling Report: Credit Card Fraud Detection

**Project**: Explainable Credit Card Fraud Detection & Risk Scoring  
**Dataset**: Kaggle ULB Credit Card Fraud Detection (`data/raw/creditcard.csv`)  
**Evaluation Date**: Day 1 Baseline Audit  
**Author**: Senior Data Scientist & ML Engineer  

---

## Executive Summary
This report documents the rigorous data validation, quality profiling, and statistical exploratory analysis conducted on the Kaggle ULB Credit Card Fraud dataset. The dataset contains **284,807 transactions** over a 48-hour recording window across **31 numerical features** (Time, V1–V28 PCA transformations, Amount, and Class).

---

## 1. Schema & Integrity Validation

| Check Metric | Expected | Observed | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Row Count** | 284,807 | **284,807** | PASS | Exact record match |
| **Column Count** | 31 | **31** | PASS | 30 continuous features + 1 binary target |
| **Data Types** | float64 (30), int64 (1) | float64 (30), int64 (1) | PASS | No corrupted type encodings |
| **Missing Values (NaN/Null)** | 0 | **0** | PASS | Zero imputation required |
| **Negative Amounts** | 0 | **0** | PASS | No negative debit anomalies |
| **Zero-Dollar Transactions ($0.00)** | Audited | **1,825** | AUDITED | 1,798 legit, **27 fraud** |
| **Duplicate Records** | Audited | **1,081** | AUDITED | 1,062 legit, **19 fraud** |

---

## 2. Deep Dive: Key Data Quality Dimensions

### A. Missing Values
- **Audit Result**: `0` missing, null, or undefined cells across all 8,829,017 data points.
- **Decision**: No synthetic imputation (mean/median/KNN) is necessary or permitted.

### B. Duplicate Rows (1,081 rows)
- **Detailed Finding**: 
  - Total exact duplicate rows (excluding the first occurrence): **1,081**.
  - **1,062 legitimate transactions** (0.37% of non-fraud).
  - **19 fraudulent transactions** (3.86% of all fraud cases).
- **Domain Context**: In high-throughput card networks, identical PCA feature vectors and amounts occur naturally due to automated retry mechanisms, recurring subscriptions, batch processing, or card testing attacks (where fraudsters test the same compromised card multiple times within seconds).
- **Engineering Decision**: **DO NOT DROP DUPLICATES**. Dropping duplicates indiscriminately would remove nearly 4% of already scarce positive fraud examples ($19/492$). The raw dataset will be preserved intact to prevent severe signal destruction.

### C. Zero-Dollar Authorizations ($0.00 Transactions)
- **Detailed Finding**: 
  - Exactly **1,825 transactions** have `Amount == 0.00`.
  - **27 of these are confirmed fraud** ($1.48\%$ fraud rate).
- **Domain Context**: In banking and merchant acquirers, zero-dollar transactions represent **card authorization pings / account verification inquiries** (e.g. AVS / CVV check when adding a card to a digital wallet). Fraudsters systematically use $0 or $1 pings to check if stolen credit card credentials are still active.
- **Engineering Decision**: Retain $0.00 transactions. Create an explicit categorical or binary indicator feature (`is_zero_auth`) during Day 3 feature engineering to capture this high-risk signal ($1.48\%$ fraud rate vs $0.17\%$ population average).

### D. Severe Class Imbalance
- **Target Distribution**:
  - Class 0 (Legitimate): **284,315 transactions (99.827%)**
  - Class 1 (Fraudulent): **492 transactions (0.173%)**
  - **Imbalance Ratio**: **~578 to 1**.
- **Financial Volume**:
  - Legitimate Transaction Volume: **\$25,102,462.04** ($99.76\%$)
  - Fraudulent Transaction Volume: **\$60,127.97** ($0.24\%$)
- **Modeling Implications**:
  - Standard accuracy is meaningless (a naive model predicting all zeros achieves $99.83\%$ accuracy while failing to detect any fraud).
  - Evaluation must strictly rely on **Precision-Recall AUC (PR-AUC / Average Precision)**, **Recall at High Precision ($\ge 0.80$)**, and **Cost-Sensitive Expected Financial Loss**.

### E. Distribution Skewness & Outliers
- **Amount Outliers**:
  - Non-fraud amounts range from \$0.00 to \$25,691.16 with a median of \$22.00 and high right-skewness (std: \$250.11).
  - Fraud amounts range from \$0.00 to \$2,125.87 with a median of \$9.25 (mean: \$122.21).
  - Because median fraud amount is small (\$9.25), fraud relies heavily on high-frequency micro-theft, punctuated by large thefts up to \$2,125.87.
- **Decision**: Standard Z-score scaling will distort distributions due to extreme positive outliers. **RobustScaler** (using median and interquartile range) or Log-transforms ($\log(Amount + 1)$) must be used in Day 3.

---

## 3. Dataset Limitations
1. **PCA Anonymization (V1–V28)**: Due to confidentiality, original cardholder identifiers, merchant categories (MCC), transaction locations (geo-coordinates), and device fingerprints are unavailable.
2. **Short Temporal Span**: The dataset covers exactly 172,792 seconds (**~48 hours / 2 days**), preventing the study of long-term seasonal trends, month-end pay cycles, or holidays.
3. **No Temporal Order per Account**: Account IDs are stripped, preventing traditional customer-level velocity aggregations (e.g., number of purchases in the last 10 minutes by cardholder).

---

## 4. Key Cleaning & Transformation Decisions for Day 2/3
1. **Raw Preservation**: Preserve `data/raw/creditcard.csv` as read-only.
2. **Parquet Storage**: Convert raw data to an optimized, snappy-compressed Parquet format for sub-second analytical queries.
3. **Feature Scaling**: Apply `RobustScaler` to `Amount` and `Time` features on training splits only.
4. **Temporal Feature Extraction**: Transform `Time` into cyclical features ($\sin / \cos$ of hour-of-day: $0-23$) to capture the diurnal drop in legitimate volume and surge in nighttime fraud rates.
