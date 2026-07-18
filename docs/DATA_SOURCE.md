# Data source and scope

## Source

This project uses the **Financial Fraud Alert Review Dataset (FiFAR)** published through Springer Nature's Figshare repository:

- DOI: <https://doi.org/10.6084/m9.figshare.28351172>
- Downloaded file: `FiFAR.zip`
- Local SHA-256: `bbd9fafb4abae6563d9d0448440b74b055c8d7945cee17822a11f9b35bf72302`
- Licence shown by the repository: CC BY

FiFAR builds on the Bank Account Fraud (BAF) dataset and adds model scores, selected alerts, synthetic fraud-analyst decisions and review-capacity scenarios.

## What the records represent

The records describe synthetic bank-account opening applications. They are not payment-card transactions and they do not represent identifiable people. The binary target identifies fraudulent and legitimate applications.

The project separates two tasks:

1. **Fraud detection:** rank applications by fraud risk.
2. **Alert review:** decide which flagged applications should be reviewed when analyst capacity is limited.

## Supplied files

| File | Role |
|---|---|
| `alert_data/Base.csv` | Base application records and fraud labels |
| `alert_data/processed_data/BAF_alert_model_score.parquet` | Scored application records supplied by the authors |
| `alert_data/processed_data/alerts.parquet` | 30,622 applications selected as alerts |
| `synthetic_experts/expert_predictions.parquet` | Decisions from 50 synthetic fraud analysts |
| `synthetic_experts/prob_of_error.parquet` | Case-level false-positive and false-negative probabilities |
| `synthetic_experts/expert_parameters.parquet` | Parameters used to generate synthetic analysts |
| `testbed/` | Team composition, batch and capacity scenarios |

## Verified source limitations

The supplied `Base.csv` contains 917,174 rows rather than the one million records described for the original BAF Base dataset. The final row is truncated: `device_fraud_count` and `month` are missing. Month 4 contains materially fewer records than the other source months and is treated as potentially incomplete.

The project therefore:

- removes only the final structurally incomplete row;
- does not impute the truncated values;
- excludes month 4 from primary temporal model comparisons;
- retains month 4 for clearly labelled descriptive checks only;
- reports results for the exact supplied archive rather than claiming results for the full BAF release.

Negative values in several duration and balance fields are documented source sentinels. They are not treated as ordinary missing values and are not silently replaced.
