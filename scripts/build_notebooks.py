"""Build the documented notebook series from reviewable cell definitions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


SETUP = code(
    """
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_ROOT = Path(os.environ.get("FIFAR_DATA_DIR", PROJECT_ROOT / "data" / "raw" / "FiFAR"))
REPORTS = PROJECT_ROOT / "reports"
IMAGES = PROJECT_ROOT / "images"

sns.set_theme(style="whitegrid")
CORAL = "#F08FA0"
TEAL = "#0E6268"
DARK = "#15262B"

if not DATA_ROOT.exists():
    raise FileNotFoundError(
        "Set FIFAR_DATA_DIR to the extracted official FiFAR directory before running this notebook."
    )
"""
)


def write(name: str, title: str, summary: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        markdown(f"# {title}\n\n{summary}"),
        markdown(
            """
## Reading guide

This notebook is part of a connected workflow. It states the decision being made, shows the supporting checks and records limitations alongside the result. Source files are never modified in place.
"""
        ),
        SETUP,
        *cells,
    ]
    for position, cell in enumerate(notebook["cells"]):
        identity = f"{name}:{position}:{cell.cell_type}:{cell.source}".encode()
        cell["id"] = hashlib.sha1(identity).hexdigest()[:12]
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nbf.write(notebook, NOTEBOOKS / name)


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    write(
        "01_data_source_and_quality.ipynb",
        "01 · Data Source and Quality",
        "Verify what was supplied, identify structural limitations and define which records are safe for each stage of the analysis.",
        [
            markdown("## 1. Source files\n\nFiFAR contains a base bank-account application table, author-generated risk scores, a selected alert population, synthetic analyst decisions and capacity scenarios. These components answer different questions and must not be mixed without checking their keys and row order."),
            code("""
paths = {
    "base": DATA_ROOT / "alert_data" / "Base.csv",
    "scores": DATA_ROOT / "alert_data" / "processed_data" / "BAF_alert_model_score.parquet",
    "alerts": DATA_ROOT / "alert_data" / "processed_data" / "alerts.parquet",
    "expert_predictions": DATA_ROOT / "synthetic_experts" / "expert_predictions.parquet",
    "expert_parameters": DATA_ROOT / "synthetic_experts" / "expert_parameters.parquet",
}
pd.DataFrame({"file": paths.keys(), "exists": [path.exists() for path in paths.values()]})
"""),
            markdown("## 2. Base schema and target"),
            code("base = pd.read_csv(paths['base'])\nbase.shape, base.head(3)"),
            code("""
schema = pd.DataFrame({
    "dtype": base.dtypes.astype(str),
    "missing": base.isna().sum(),
    "unique": base.nunique(dropna=False),
})
schema
"""),
            markdown("## 3. Structural completeness\n\nA value of `-1` is a documented source sentinel in several fields. It is not the same as a truncated CSV row and must not be replaced without a field-specific decision."),
            code("""
quality = pd.Series({
    "rows": len(base),
    "columns": base.shape[1],
    "duplicate_rows": base.duplicated().sum(),
    "missing_cells": base.isna().sum().sum(),
    "incomplete_rows": base.isna().any(axis=1).sum(),
    "fraud_cases": base["fraud_bool"].sum(),
    "fraud_rate": base["fraud_bool"].mean(),
})
quality
"""),
            code("base.loc[base.isna().any(axis=1)]"),
            markdown("**Decision.** Remove only the final structurally incomplete row. Preserve documented sentinels and report their frequency separately."),
            code("base_complete = base.loc[~base.isna().any(axis=1)].copy()\nbase_complete.shape"),
            markdown("## 4. Monthly completeness"),
            code("""
monthly = base_complete.groupby("month")["fraud_bool"].agg(
    applications="size", fraud_cases="sum", fraud_rate="mean"
)
monthly
"""),
            code("""
fig, left = plt.subplots(figsize=(11, 5))
right = left.twinx()
left.bar(monthly.index, monthly["applications"], color=TEAL, alpha=.82)
right.plot(monthly.index, monthly["fraud_rate"] * 100, color=CORAL, marker="o", linewidth=2.5)
left.axvspan(3.65, 4.35, color=CORAL, alpha=.12)
left.set(xlabel="Source month", ylabel="Applications", title="Volume and observed fraud rate")
right.set_ylabel("Fraud rate (%)")
plt.show()
"""),
            markdown("Month 4 has substantially lower volume and an implausibly elevated observed fraud rate. Because the file terminates during this month, it is excluded from primary temporal comparisons rather than treated as genuine distribution shift."),
            markdown("## 5. Related table alignment"),
            code("""
alerts = pd.read_parquet(paths["alerts"])
scores = pd.read_parquet(paths["scores"])
experts = pd.read_parquet(paths["expert_predictions"])
pd.DataFrame({
    "table": ["base", "author scores", "alerts", "expert predictions"],
    "rows": [len(base), len(scores), len(alerts), len(experts)],
    "columns": [base.shape[1], scores.shape[1], alerts.shape[1], experts.shape[1]],
})
"""),
            markdown("The expert-prediction table has the same row count as the alerts table. Their row alignment is used only after confirming this invariant. The scored table is a separate author-provided population and is not joined by row number to the base CSV."),
            markdown("## Conclusion\n\nThe archive supports a fraud-ranking study and a downstream alert-review study. The incomplete base month is a material limitation, but it can be isolated without fabricating records or values."),
        ],
    )

    write(
        "02_fraud_pattern_analysis.ipynb",
        "02 · Fraud Pattern Analysis",
        "Describe rarity and temporal variation before building a model, while avoiding causal claims from synthetic data.",
        [
            code("base = pd.read_csv(DATA_ROOT / 'alert_data' / 'Base.csv').dropna().copy()"),
            markdown("## 1. Class balance"),
            code("base['fraud_bool'].value_counts().rename(index={0: 'Legitimate', 1: 'Fraud'}).to_frame('applications')"),
            code("""
fraud_rate = base["fraud_bool"].mean()
print(f"Observed fraud rate: {fraud_rate:.2%}")
print(f"A classifier predicting every case as legitimate would be {1-fraud_rate:.2%} accurate and operationally useless.")
"""),
            markdown("## 2. Change over source months"),
            code("""
month_profile = base.groupby("month")["fraud_bool"].agg(applications="size", fraud_cases="sum", fraud_rate="mean")
month_profile.drop(index=4).assign(fraud_rate=lambda frame: (frame["fraud_rate"] * 100).round(2))
"""),
            markdown("Month 4 is shown only in the data-quality notebook. It is removed here because the source file is incomplete during that period."),
            markdown("## 3. Numeric feature contrasts"),
            code("""
numeric = [
    "name_email_similarity", "days_since_request", "credit_risk_score",
    "proposed_credit_limit", "session_length_in_minutes", "velocity_24h",
]
contrasts = base.groupby("fraud_bool")[numeric].median().T
contrasts.columns = ["legitimate_median", "fraud_median"]
contrasts["difference"] = contrasts["fraud_median"] - contrasts["legitimate_median"]
contrasts
"""),
            code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for axis, column in zip(axes.ravel(), ["name_email_similarity", "credit_risk_score", "proposed_credit_limit", "session_length_in_minutes"]):
    sample = base.sample(min(120_000, len(base)), random_state=42)
    sns.boxplot(data=sample, x="fraud_bool", y=column, ax=axis, color=CORAL, showfliers=False)
    axis.set(xlabel="Fraud label", title=column.replace("_", " ").title())
plt.tight_layout()
plt.show()
"""),
            markdown("The plots describe association, not cause. The records are synthetic and several variables are encoded or normalised, so values should not be translated into real customer behaviour without documentation."),
            markdown("## 4. Categorical profiles"),
            code("""
def category_profile(column):
    return base.groupby(column, observed=True)["fraud_bool"].agg(applications="size", fraud_cases="sum", fraud_rate="mean").sort_values("fraud_rate", ascending=False)

category_profile("device_os")
"""),
            code("category_profile('payment_type')"),
            code("category_profile('source')"),
            markdown("## 5. Sentinel values"),
            code("""
sentinel_columns = ["prev_address_months_count", "current_address_months_count", "bank_months_count", "device_fraud_count"]
pd.DataFrame({
    "sentinel_minus_one": [(base[c] == -1).sum() for c in sentinel_columns],
    "share": [(base[c] == -1).mean() for c in sentinel_columns],
}, index=sentinel_columns)
"""),
            markdown("**Modelling implication.** Sentinel values remain distinct. Median imputation is reserved for genuine missing values and is fitted only on the training months."),
            markdown("## Conclusion\n\nFraud is rare, changes across time and is associated with several application and device characteristics. Those patterns justify temporal validation and capacity-based metrics, but not causal interpretation."),
        ],
    )

    write(
        "03_feature_policy_and_preparation.ipynb",
        "03 · Feature Policy and Preparation",
        "Define what the model may use, what remains available only for audit and how temporal leakage is prevented.",
        [
            code("base = pd.read_csv(DATA_ROOT / 'alert_data' / 'Base.csv').dropna().copy()"),
            markdown("## 1. Feature roles"),
            code("""
target = "fraud_bool"
time_column = "month"
audit_only = ["income", "customer_age", "employment_status", "housing_status"]
categorical = ["payment_type", "source", "device_os"]
pd.Series({
    "target": target,
    "time_column": time_column,
    "audit_only": audit_only,
    "model_categorical": categorical,
})
"""),
            markdown("Income, age, employment and housing are excluded from the primary model and retained for subgroup auditing. This does not remove all possible proxies and must not be described as proof of fairness."),
            markdown("## 2. Temporal split"),
            code("""
train = base[base.month.isin([0, 1, 2, 3])].copy()
validation = base[base.month.isin([5, 6])].copy()
test = base[base.month.eq(7)].copy()
pd.DataFrame({
    "split": ["train", "validation", "test"],
    "rows": [len(train), len(validation), len(test)],
    "fraud_rate": [train.fraud_bool.mean(), validation.fraud_bool.mean(), test.fraud_bool.mean()],
})
"""),
            markdown("Month 4 is not reassigned. The test month is not used for preprocessing, model selection or threshold selection."),
            markdown("## 3. Candidate feature matrix"),
            code("""
excluded = [target, time_column, *audit_only]
model_columns = [column for column in base.columns if column not in excluded]
pd.DataFrame({"feature": model_columns, "dtype": base[model_columns].dtypes.astype(str).values})
"""),
            markdown("## 4. Preprocessing boundaries"),
            code("""
from fraud_detection.modelling import build_logistic_pipeline, build_histogram_pipeline

logistic_pipeline = build_logistic_pipeline(train)
tree_pipeline = build_histogram_pipeline(train)
logistic_pipeline
"""),
            markdown("Both pipelines fit imputers and encoders on training data only. The logistic model standardises numeric fields and one-hot encodes categories. The tree model uses ordinal codes for categories and leaves numeric scales unchanged."),
            markdown("## 5. Leakage register"),
            code("""
pd.DataFrame([
    ["model_score", "Excluded", "Author-provided prediction would leak an existing model into a new model"],
    ["fraud_bool", "Target only", "Outcome label"],
    ["month", "Split only", "Prevents random mixing across time"],
    ["audit-only fields", "Evaluation only", "Reserved for subgroup analysis"],
], columns=["field", "policy", "reason"])
"""),
            markdown("## 6. Reproducibility checks"),
            code("""
assert set(train.month.unique()) == {0, 1, 2, 3}
assert set(validation.month.unique()) == {5, 6}
assert set(test.month.unique()) == {7}
assert "model_score" not in model_columns
assert not set(audit_only).intersection(model_columns)
print("Feature and split checks passed.")
"""),
            markdown("## Conclusion\n\nThe feature policy is intentionally conservative. It separates model inputs, temporal controls and audit fields before any model is fitted."),
        ],
    )

    write(
        "04_model_baselines.ipynb",
        "04 · Model Baselines",
        "Establish simple reference points before evaluating a more flexible ranking model.",
        [
            code("metrics = json.loads((REPORTS / 'model_metrics.json').read_text())"),
            markdown("## 1. Why accuracy is not a baseline\n\nFraud is close to one percent of the supplied base. Predicting every application as legitimate would appear accurate while recovering no fraud."),
            code("audit = json.loads((REPORTS / 'data_audit.json').read_text())\naudit['base']"),
            markdown("## 2. Logistic regression\n\nClass-weighted logistic regression provides an interpretable ranking baseline. It is not expected to capture complex interactions, but a more flexible model should demonstrate a clear benefit over it."),
            code("pd.Series(metrics['models']['logistic_regression']['validation'])"),
            code("pd.Series(metrics['models']['logistic_regression']['test'])"),
            markdown("## 3. Capacity view"),
            code("""
log_capacity = pd.DataFrame(metrics["models"]["logistic_regression"]["test_capacity"])
log_capacity[["review_share", "review_capacity", "fraud_captured", "precision_at_capacity", "recall_at_capacity"]]
"""),
            code("""
plt.figure(figsize=(9, 5))
plt.plot(log_capacity.review_share * 100, log_capacity.recall_at_capacity * 100, color=TEAL, marker="o", linewidth=2.5)
plt.xlabel("Applications reviewed (%)")
plt.ylabel("Fraud recovered (%)")
plt.title("Logistic baseline under fixed review capacity")
plt.show()
"""),
            markdown("At three percent review capacity, the logistic model recovers 438 of 1,428 fraud cases in the final month. This becomes the minimum useful benchmark for model comparison."),
            markdown("## Conclusion\n\nThe logistic baseline ranks fraud meaningfully, but its performance leaves room for a model that captures nonlinear relationships without relying on audit-only fields."),
        ],
    )

    write(
        "05_model_comparison.ipynb",
        "05 · Model Comparison",
        "Compare the baseline and gradient-boosting model using validation choices and an untouched final month.",
        [
            code("metrics = json.loads((REPORTS / 'model_metrics.json').read_text())"),
            code("""
rows = []
for model_name, model_result in metrics["models"].items():
    for split in ["validation", "test"]:
        rows.append({"model": model_name, "split": split, **model_result[split]})
comparison = pd.DataFrame(rows)
comparison
"""),
            markdown("## 1. Ranking performance"),
            code("comparison.pivot(index='model', columns='split', values=['average_precision', 'roc_auc'])"),
            markdown("Average precision is the primary ranking metric because it reflects performance on the rare positive class. ROC AUC is reported as a secondary measure."),
            markdown("## 2. Review-capacity comparison"),
            code("""
capacity = []
for model_name, result in metrics["models"].items():
    for row in result["test_capacity"]:
        capacity.append({"model": model_name, **row})
capacity = pd.DataFrame(capacity)
capacity
"""),
            code("""
plt.figure(figsize=(10, 6))
for model_name, group in capacity.groupby("model"):
    plt.plot(group.review_share * 100, group.recall_at_capacity * 100, marker="o", linewidth=2.5, label=model_name.replace("_", " ").title())
plt.xlabel("Applications reviewed (%)")
plt.ylabel("Fraud recovered (%)")
plt.title("Final-month recovery by review capacity")
plt.legend(frameon=False)
plt.show()
"""),
            markdown("## 3. Selection decision"),
            code("""
three_percent = capacity[capacity.review_share.eq(.03)].set_index("model")
three_percent[["fraud_captured", "precision_at_capacity", "recall_at_capacity"]]
"""),
            markdown("Histogram gradient boosting is selected because it improves validation average precision and fraud recovery at every tested capacity. Its final-month results are reported once, after that selection."),
            markdown("## Limitation\n\nThe comparison covers two well-defined baselines, not an exhaustive model search. Additional models should be added only if they improve the validation result and preserve reproducibility."),
        ],
    )

    write(
        "06_threshold_and_capacity.ipynb",
        "06 · Threshold and Review Capacity",
        "Translate model scores into operational choices and quantify the trade-off between fraud recovered and investigations created.",
        [
            code("metrics = json.loads((REPORTS / 'model_metrics.json').read_text())\nselected = metrics['models']['hist_gradient_boosting']"),
            markdown("## 1. Two different decisions\n\nA score threshold fixes a risk boundary. A capacity rule reviews the highest-risk cases until the available queue is full. In a changing population, the two policies do not necessarily produce the same volume."),
            code("pd.Series({'validation_f2_threshold': selected['validation_threshold_f2'], **selected['test']})"),
            markdown("## 2. Fixed-capacity scenarios"),
            code("capacity = pd.DataFrame(selected['test_capacity'])\ncapacity"),
            code("""
capacity["fraud_missed"] = 1428 - capacity["fraud_captured"]
capacity["false_positive_reviews"] = capacity["review_capacity"] - capacity["fraud_captured"]
capacity[["review_share", "review_capacity", "fraud_captured", "fraud_missed", "false_positive_reviews"]]
"""),
            code("""
fig, axis = plt.subplots(figsize=(10, 6))
axis.plot(capacity.review_share * 100, capacity.recall_at_capacity * 100, color=CORAL, marker="o", linewidth=2.5, label="Fraud recovered")
axis.plot(capacity.review_share * 100, capacity.precision_at_capacity * 100, color=TEAL, marker="o", linewidth=2.5, label="Queue precision")
axis.set(xlabel="Applications reviewed (%)", ylabel="Percent", title="Recovery and precision move in different directions")
axis.legend(frameon=False)
plt.show()
"""),
            markdown("Increasing capacity recovers more fraud but lowers the concentration of fraud inside the queue. A business decision therefore requires both the cost of a missed fraud and the cost of a manual review; those costs are not supplied and are not invented here."),
            markdown("## 3. Recommended reporting point\n\nThe portfolio uses three percent as a transparent comparison scenario, not as a production recommendation. At this capacity, 2,905 applications are reviewed, 533 fraud cases are captured and 2,372 reviews are false positives."),
            markdown("## Conclusion\n\nA model metric is not an operating policy. The review-capacity analysis makes the workload and missed-fraud trade-off visible before any claim of business value."),
        ],
    )

    write(
        "07_analyst_review.ipynb",
        "07 · Analyst Review",
        "Measure variation across the 50 synthetic fraud analysts and compare capacity-aware assignment policies on a later alert month.",
        [
            code("""
alerts = pd.read_parquet(DATA_ROOT / "alert_data" / "processed_data" / "alerts.parquet")
predictions = pd.read_parquet(DATA_ROOT / "synthetic_experts" / "expert_predictions.parquet")
parameters = pd.read_parquet(DATA_ROOT / "synthetic_experts" / "expert_parameters.parquet")
assert alerts.index.is_unique and predictions.index.is_unique
assert set(alerts.index) == set(predictions.index)
predictions = predictions.reindex(alerts.index)
assert alerts.index.equals(predictions.index)
"""),
            markdown("## 1. Analyst-level performance"),
            code("""
truth = alerts["fraud_bool"].to_numpy()
analyst_rows = []
for analyst in predictions.columns:
    decision = predictions[analyst].to_numpy()
    tp = int(((decision == 1) & (truth == 1)).sum())
    fp = int(((decision == 1) & (truth == 0)).sum())
    fn = int(((decision == 0) & (truth == 1)).sum())
    tn = int(((decision == 0) & (truth == 0)).sum())
    analyst_rows.append({
        "analyst": analyst,
        "accuracy": (tp + tn) / len(truth),
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": tp / (tp + fn) if tp + fn else 0,
        "positive_rate": (tp + fp) / len(truth),
    })
analysts = pd.DataFrame(analyst_rows).sort_values("accuracy", ascending=False)
analysts.head(10)
"""),
            code("analysts.describe().T"),
            markdown("Accuracy varies widely across the synthetic team. It must be interpreted with precision, recall and decision rate because the alert population remains imbalanced."),
            code("""
fig, axis = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=analysts, x="recall", y="precision", size="positive_rate", color=CORAL, alpha=.75, ax=axis)
axis.set(title="Synthetic analysts make different precision–recall trade-offs", xlabel="Recall", ylabel="Precision")
plt.show()
"""),
            markdown("## 2. Capacity scenarios"),
            code("""
capacity_files = sorted((DATA_ROOT / "testbed" / "test").glob("*/capacity.csv"))
batch_files = sorted((DATA_ROOT / "testbed" / "test").glob("*/batches.csv"))
pd.Series({"test_capacity_files": len(capacity_files), "test_batch_files": len(batch_files)})
"""),
            code("pd.read_csv(capacity_files[0]).head(), pd.read_csv(batch_files[0]).head()"),
            markdown("## 3. Historical estimation and final test\n\nAnalyst skill is estimated from alert months 3–6. Month 7 remains outside this estimation and is used to compare 25 supplied team scenarios. Score bands are fitted on the historical alerts only."),
            code("review = json.loads((REPORTS / 'review_strategy_metrics.json').read_text())\npd.Series(review['data_split'])"),
            markdown("## 4. Assignment policies\n\n- **Random capacity:** distributes cases randomly within each active analyst's limit.\n- **Global skill:** prioritises analysts with the strongest historical correctness where review is expected to improve on the screening decision.\n- **Risk-band specialist:** estimates each analyst's correctness within ten historical score bands, with smoothing towards their overall result.\n\nEvery policy respects the supplied analyst capacities. Unreviewed alerts retain the screening decision."),
            code("""
strategy = pd.DataFrame(review["strategy_summary"])
display_columns = [
    "strategy", "scenarios", "mean_accuracy", "mean_precision", "mean_recall",
    "mean_false_positive", "mean_false_negative", "mean_human_reviews",
]
strategy[display_columns].sort_values("mean_accuracy", ascending=False)
"""),
            code("""
plot_data = strategy.melt(
    id_vars="strategy",
    value_vars=["mean_accuracy", "mean_precision", "mean_recall"],
    var_name="metric",
    value_name="value",
)
plot_data["metric"] = plot_data["metric"].str.replace("mean_", "", regex=False).str.title()
plot_data["policy"] = plot_data["strategy"].str.replace("_", " ", regex=False).str.title()
fig, axis = plt.subplots(figsize=(11, 6))
sns.barplot(data=plot_data, x="policy", y="value", hue="metric", palette=[TEAL, CORAL, "#728489"], ax=axis)
axis.set(title="Assignment policies create different review trade-offs", xlabel="Assignment policy", ylabel="Mean across 25 scenarios")
axis.legend(title=None, frameon=False)
plt.show()
"""),
            markdown("## 5. Result\n\nRisk-band assignment achieves the highest mean accuracy and precision, while random assignment retains slightly more fraud cases. The gain over global-skill assignment is small. This is a trade-off, not evidence that a single policy is best for every operating cost."),
            code("""
random_row = strategy.set_index("strategy").loc["random_capacity"]
specialist_row = strategy.set_index("strategy").loc["risk_band_specialist"]
pd.Series({
    "accuracy_change_vs_random_pp": (specialist_row.mean_accuracy - random_row.mean_accuracy) * 100,
    "precision_change_vs_random_pp": (specialist_row.mean_precision - random_row.mean_precision) * 100,
    "recall_change_vs_random_pp": (specialist_row.mean_recall - random_row.mean_recall) * 100,
    "mean_false_positives_avoided": random_row.mean_false_positive - specialist_row.mean_false_positive,
})
"""),
            markdown("## Conclusion\n\nCapacity and analyst selection materially change the final alert decisions. Historical specialisation can reduce unnecessary positive decisions, but the accompanying recall loss must be assessed against fraud cost and review policy."),
        ],
    )

    write(
        "08_stability_fairness_and_findings.ipynb",
        "08 · Stability, Subgroup Audit and Findings",
        "Bring model, capacity and source limitations together without presenting synthetic results as production evidence.",
        [
            code("base = pd.read_csv(DATA_ROOT / 'alert_data' / 'Base.csv').dropna().copy()\nmetrics = json.loads((REPORTS / 'model_metrics.json').read_text())\nreview = json.loads((REPORTS / 'review_strategy_metrics.json').read_text())"),
            markdown("## 1. Temporal stability"),
            code("""
monthly = base[~base.month.eq(4)].groupby("month")["fraud_bool"].agg(applications="size", fraud_cases="sum", fraud_rate="mean")
monthly.assign(fraud_rate_percent=lambda frame: (frame["fraud_rate"] * 100).round(2))
"""),
            markdown("The observed fraud rate changes across the supplied months. This supports temporal monitoring and makes random-only validation inappropriate."),
            markdown("## 2. Audit fields"),
            code("""
audit_fields = ["income", "customer_age", "employment_status", "housing_status"]
pd.DataFrame({
    "field": audit_fields,
    "used_by_primary_model": [False] * len(audit_fields),
    "retained_for_subgroup_audit": [True] * len(audit_fields),
})
"""),
            markdown("Excluding these fields does not guarantee fairness. Remaining features may act as proxies, and the synthetic dataset cannot establish legal or regulatory suitability."),
            code("""
age_profile = base.groupby("customer_age")["fraud_bool"].agg(applications="size", fraud_cases="sum", fraud_rate="mean")
age_profile.assign(fraud_rate_percent=lambda frame: (frame["fraud_rate"] * 100).round(2))
"""),
            code("""
employment_profile = base.groupby("employment_status")["fraud_bool"].agg(applications="size", fraud_cases="sum", fraud_rate="mean")
employment_profile.assign(fraud_rate_percent=lambda frame: (frame["fraud_rate"] * 100).round(2))
"""),
            markdown("These are descriptive target distributions, not model fairness metrics. Full subgroup evaluation requires saved test predictions, minimum-group support rules and uncertainty intervals."),
            markdown("## 3. Consolidated final-month result"),
            code("""
selected = metrics["models"]["hist_gradient_boosting"]
three_percent = next(row for row in selected["test_capacity"] if row["review_share"] == .03)
pd.Series({
    "test_applications": metrics["split"]["test_rows"],
    "average_precision": selected["test"]["average_precision"],
    "roc_auc": selected["test"]["roc_auc"],
    **three_percent,
})
"""),
            markdown("## 4. Alert-review result"),
            code("pd.DataFrame(review['strategy_summary']).sort_values('mean_accuracy', ascending=False)"),
            markdown("## 5. Findings\n\n- The supplied application target is rare, so accuracy is not informative for the ranking model.\n- Temporal evaluation exposes changing fraud prevalence.\n- Gradient boosting improves ranking and fraud recovery over logistic regression.\n- At three percent capacity, the queue captures 533 of 1,428 fraud cases.\n- That queue still contains 2,372 legitimate applications and misses 895 fraud cases.\n- Across 25 test teams, risk-band assignment reduces mean false positives by about 180 compared with random capacity allocation, while recall falls by about 1.7 percentage points.\n- Analyst variation makes assignment and workload part of the detection problem."),
            markdown("## 6. Limitations\n\n- All applications and analysts are synthetic.\n- The supplied base CSV is truncated during month 4.\n- The current model comparison is deliberately limited.\n- No monetary loss or investigation cost is supplied.\n- Protected-field exclusion does not prove fairness.\n- Analyst skill is estimated from a short historical window and may change.\n- Results do not demonstrate production or regulatory readiness."),
            markdown("## Recommendation\n\nUse the model as a ranking component, not an automatic rejection rule. Monitor queue precision, fraud recovery, monthly drift, subgroup behaviour and analyst-assignment outcomes. Select the assignment objective only after defining the relative cost of false positives and missed fraud."),
        ],
    )


if __name__ == "__main__":
    main()
