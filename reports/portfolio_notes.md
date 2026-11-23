# Portfolio notes

## GitHub About

**Description** · Explainable credit card fraud detection research prototype with calibrated Random Forest scoring, chronological evaluation, SHAP, and a Streamlit dashboard.

**Suggested topics** · `fraud-detection`, `machine-learning`, `data-science`, `credit-card-fraud`, `imbalanced-classification`, `model-calibration`, `shap`, `streamlit`, `duckdb`, `python`

## Internship resume bullets

- Built an explainable fraud detection prototype on 284,807 ULB transactions, benchmarking logistic regression, Random Forest, and XGBoost with chronological splits; the selected calibrated Random Forest achieved **0.791712 average precision** on 56,962 final test transactions.
- Developed a four-page Streamlit dashboard with transaction scoring, TreeSHAP contributions, and CSV batch screening; at the validation-selected **0.390** review threshold, final test evaluation caught **53 of 75 fraud cases** with **6 false positives**.

## Interview Q&A

**1. Why average precision?**
Fraud is rare, so accuracy can stay high while fraud goes undetected. Average precision summarizes precision across recall levels and responds directly to false review flags. Final test average precision was 0.791712.

**2. How did you prevent training leakage?**
The project used chronological 60/20/20 partitions. The calibrated RF fit on the earlier 75% of training rows; a frozen estimator received sigmoid calibration on the later 25%. Validation selected the model and threshold. Final test was scored once after selection. Earlier EDA and duplicate audits inspected the full dataset.

**3. Why Random Forest when XGBoost was competitive?**
XGBoost was a benchmark finalist. The calibrated RF achieved validation F1 of 0.8000 at 0.390 and Brier score of 0.000325. Selection used the validation operating point and calibration, without using final test metrics.

**4. What does the SHAP chart explain?**
TreeSHAP attributes the underlying RF output to the 30 input features. Sigmoid calibration changes the probability scale, so the SHAP values do not sum to the displayed calibrated probability. V1–V28 are anonymized components, not merchant or customer fields.

**5. What would operational use require?**
Longer-horizon and external validation, drift monitoring, access controls, investigation workflows, measured review costs, and governance review. The dashboard is a research demonstration; its review flag does not change a payment's status.

## Screenshots to capture

No dashboard screenshot files are present. Capture real views of:

1. Data Intelligence with class distribution and transaction charts.
2. Model Performance with validation comparison, PR curves, and calibration chart.
3. Fraud Prediction with a validation fraud sample, calibrated score, review decision, and TreeSHAP chart.
4. Batch Screening after uploading the bundled 100-row CSV, showing the scored table and download control.

Save approved captures under `reports/screenshots/` and link them from the README after checking for sensitive local paths or unrelated browser content.
