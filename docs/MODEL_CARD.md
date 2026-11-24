# Model card · Fraud Risk Intelligence

**Model** — sigmoid-calibrated Random Forest. **Decision** — suggest human review when the calibrated fraud probability reaches the frozen validation threshold of **0.390**. The dashboard uses the full-precision threshold stored in its metadata. [Threshold metadata](../data/demo/dashboard_summary.json)

## Intended use

Use the dashboard to explore fraud screening on ULB-compatible, anonymized research transactions. An analyst can rank records, inspect a score and prioritize a limited review queue. Exclude automated payment holds, customer eligibility decisions and production loss forecasts from this model's uses. A flag conveys a review suggestion. Establishing fraud requires independent evidence.

## Data, fitting and selection

The ULB dataset contains 284,807 transactions over approximately two days, including 492 labeled frauds. Inputs are elapsed time, transaction amount and 28 anonymized PCA components. `Class` supplies the research outcome. Optional uploaded identifiers serve only to match scores back to records. [Data quality report](../reports/data_quality_report.md)

| Chronological partition | Records | Role |
| --- | ---: | --- |
| Earlier 75% of training | 128,163 | Fit preprocessing and Random Forest |
| Later 25% of training | 42,721 | Fit the sigmoid calibrator on the frozen forest's outputs |
| Validation | 56,961 | Compare candidates and select the review threshold by F1 |
| Final test | 56,962 | Evaluate the frozen model and policy once |

The pipeline robust-scales `Time`, applies `log1p` and robust scaling to `Amount`, and passes V1–V28 through. The fit and calibration records occupy disjoint positions within the first 60% of the dataset. The fitted preprocessor transforms the later records. Scikit-learn's `FrozenEstimator` holds the fitted forest fixed during sigmoid calibration. Logistic regression provided a baseline and XGBoost was a benchmark finalist. The calibrated Random Forest won on validation F1. Earlier exploratory profiling and duplicate audits covered the full dataset. [Validation summaries](../data/demo/dashboard_summary.json) · [Evaluation report](../reports/final_test_metrics.json)

## Frozen evaluation

The following aggregates come from the saved evaluation. Precision, recall and F1 use the same fixed review policy in both columns. Average precision summarizes the precision–recall curve across thresholds. Lower Brier score and log loss indicate better probability predictions. [Full-precision results and model SHA-256](../reports/final_test_metrics.json)

| Metric | Validation | Final test |
| --- | ---: | ---: |
| Labeled frauds | 57 | 75 |
| Average precision | 0.756553 | 0.791712 |
| ROC-AUC | 0.965722 | 0.966213 |
| Precision | 0.930233 | 0.898305 |
| Recall | 0.701754 | 0.706667 |
| F1 | 0.800000 | 0.791045 |
| Brier score | 0.000325 | 0.000419 |
| Log loss | 0.002359 | 0.002875 |

Of the 59 final-test records flagged for review, 53 were fraud and 6 were legitimate. Of the records below threshold, 22 were fraud and 56,881 were legitimate. The six false positives represent extra analyst work, and the 22 false negatives show what a review process relying solely on this signal would miss. A top-K queue can leave some flagged records outside the displayed list. Financial costs in the evaluation are illustrative assumptions. [Aggregate confusion matrices](../reports/final_test_metrics.json)

## Reading a score

| Calibrated probability | Display tier | Review suggestion |
| --- | --- | --- |
| Below 0.10 | Low | No flag |
| 0.10 to below the review threshold | Medium | No flag |
| Review threshold to below 0.50 | Medium | Flag |
| 0.50 and above | High | Flag |

TreeSHAP attributes the underlying forest's **pre-calibration** fraud output to the transformed features. Adding contributions to the explainer baseline reconstructs that forest output. The displayed probability includes a subsequent sigmoid transformation. The anonymized PCA axes have no published business interpretation here, and the attributions cannot establish why a customer acted or whether fraud occurred.

## Generalization, privacy and ethics

- The two-day window and 75 positive holdout cases limit precision and recall estimates. The project provides no external validation across institutions, time periods or customer groups. Changes in fraud prevalence and behavior can affect both calibration and review workload.
- The 100-row demonstration CSV deliberately includes ten fraud examples selected from validation. It omits labels. Use it to exercise the interface. Neither its review rate nor its score distribution estimates operational accuracy or fraud prevalence. [Demo sample metadata](../data/demo/dashboard_summary.json)
- Use anonymous test records and non-sensitive reference IDs only in the public app. The schema checks numeric inputs but cannot verify that uploaded values use the original PCA transformation or detect personal information in optional fields.
- Subgroup error rates, fairness and real financial impact have not been established. The research dashboard provides no production access controls, drift monitoring or case-management audit trail. A customer-facing decision would require separate validation and governance.
