"""Build small, publishable aggregates for the interactive dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.data_io import align_case_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path, default=Path("dashboard/data"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    audit = json.loads((args.reports / "data_audit.json").read_text(encoding="utf-8"))
    metrics = json.loads((args.reports / "model_metrics.json").read_text(encoding="utf-8"))
    pd.DataFrame(audit["base_by_month"]).to_csv(args.output / "monthly_summary.csv", index=False)

    capacity_rows = []
    for model_name, result in metrics["models"].items():
        for split_name in ("validation_capacity", "test_capacity"):
            for row in result[split_name]:
                capacity_rows.append(
                    {
                        "model": model_name,
                        "split": split_name.removesuffix("_capacity"),
                        **row,
                    }
                )
    pd.DataFrame(capacity_rows).to_csv(args.output / "capacity_summary.csv", index=False)

    alerts = pd.read_parquet(args.source / "alert_data" / "processed_data" / "alerts.parquet")
    predictions = pd.read_parquet(args.source / "synthetic_experts" / "expert_predictions.parquet")
    predictions = align_case_index(alerts, predictions)
    truth = alerts["fraud_bool"].to_numpy()
    analysts = []
    for analyst in predictions.columns:
        decision = predictions[analyst].to_numpy()
        true_positive = int(((decision == 1) & (truth == 1)).sum())
        false_positive = int(((decision == 1) & (truth == 0)).sum())
        false_negative = int(((decision == 0) & (truth == 1)).sum())
        true_negative = int(((decision == 0) & (truth == 0)).sum())
        analysts.append(
            {
                "analyst": analyst,
                "accuracy": (true_positive + true_negative) / len(truth),
                "precision": true_positive / (true_positive + false_positive),
                "recall": true_positive / (true_positive + false_negative),
                "positive_rate": (true_positive + false_positive) / len(truth),
            }
        )
    pd.DataFrame(analysts).to_csv(args.output / "analyst_summary.csv", index=False)

    review = json.loads(
        (args.reports / "review_strategy_metrics.json").read_text(encoding="utf-8")
    )
    model_only = pd.DataFrame([review["model_only"]])
    strategy_summary = pd.DataFrame(review["strategy_summary"])
    strategy_summary.to_csv(args.output / "review_strategy_summary.csv", index=False)
    model_only.to_csv(args.output / "review_model_only.csv", index=False)


if __name__ == "__main__":
    main()
