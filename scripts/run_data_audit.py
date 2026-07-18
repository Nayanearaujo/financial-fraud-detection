"""Create reproducible data-quality summaries from the official archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.data_io import align_case_index, load_base, load_parquet
from fraud_detection.validation import audit_base, monthly_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Extracted FiFAR directory")
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_base(args.source / "alert_data" / "Base.csv")
    alerts = load_parquet(args.source / "alert_data" / "processed_data" / "alerts.parquet")
    scores = load_parquet(args.source / "alert_data" / "processed_data" / "BAF_alert_model_score.parquet")
    expert_predictions = load_parquet(args.source / "synthetic_experts" / "expert_predictions.parquet")

    expert_predictions = align_case_index(alerts, expert_predictions)
    expert_accuracy = expert_predictions.eq(alerts["fraud_bool"], axis=0).mean()
    report = {
        "base": audit_base(base).to_dict(),
        "base_by_month": monthly_summary(base.dropna()).to_dict(orient="records"),
        "alerts": {
            "rows": len(alerts),
            "fraud_cases": int(alerts["fraud_bool"].sum()),
            "fraud_rate": float(alerts["fraud_bool"].mean()),
            "share_of_supplied_base": float(len(alerts) / len(base)),
            "share_of_supplied_base_fraud_captured": float(
                alerts["fraud_bool"].sum() / base["fraud_bool"].sum()
            ),
        },
        "author_scored_applications": {
            "rows": len(scores),
            "fraud_cases": int(scores["fraud_bool"].sum()),
            "fraud_rate": float(scores["fraud_bool"].mean()),
        },
        "synthetic_experts": {
            "count": expert_predictions.shape[1],
            "mean_accuracy_on_alerts": float(expert_accuracy.mean()),
            "minimum_accuracy_on_alerts": float(expert_accuracy.min()),
            "maximum_accuracy_on_alerts": float(expert_accuracy.max()),
            "best_expert": str(expert_accuracy.idxmax()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
