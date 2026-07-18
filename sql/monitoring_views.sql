CREATE OR REPLACE VIEW monthly_fraud_monitoring AS
SELECT
    CAST(month AS INTEGER) AS source_month,
    CAST(applications AS BIGINT) AS applications,
    CAST(fraud_cases AS BIGINT) AS fraud_cases,
    CAST(fraud_rate AS DOUBLE) AS fraud_rate
FROM read_csv_auto('dashboard/data/monthly_summary.csv', header = true)
WHERE CAST(month AS INTEGER) <> 4;

CREATE OR REPLACE VIEW model_capacity_comparison AS
SELECT
    split,
    CAST(review_share AS DOUBLE) AS review_share,
    MAX(CASE WHEN model = 'logistic_regression' THEN fraud_captured END) AS logistic_fraud_captured,
    MAX(CASE WHEN model = 'hist_gradient_boosting' THEN fraud_captured END) AS boosting_fraud_captured,
    MAX(CASE WHEN model = 'hist_gradient_boosting' THEN precision_at_capacity END) AS boosting_precision,
    MAX(CASE WHEN model = 'hist_gradient_boosting' THEN recall_at_capacity END) AS boosting_recall
FROM read_csv_auto('dashboard/data/capacity_summary.csv', header = true)
GROUP BY split, review_share;

CREATE OR REPLACE VIEW review_strategy_monitoring AS
SELECT
    strategy,
    CAST(scenarios AS INTEGER) AS scenarios,
    CAST(mean_accuracy AS DOUBLE) AS mean_accuracy,
    CAST(mean_precision AS DOUBLE) AS mean_precision,
    CAST(mean_recall AS DOUBLE) AS mean_recall,
    CAST(mean_false_positive AS DOUBLE) AS mean_false_positive,
    CAST(mean_false_negative AS DOUBLE) AS mean_false_negative
FROM read_csv_auto('dashboard/data/review_strategy_summary.csv', header = true);
