"""Compare model-only and capacity-aware alert-review strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fraud_detection.assignment import (
    apply_score_bands,
    evaluate_assignments,
    greedy_advantage_assignment,
    historical_skill_tables,
    random_assignment,
    score_band_edges,
)
from fraud_detection.data_io import align_case_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/review_strategy_metrics.json"))
    parser.add_argument("--assignments", type=Path, default=Path("reports/review_assignments.parquet"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alerts = pd.read_parquet(args.source / "alert_data" / "processed_data" / "alerts.parquet")
    predictions = pd.read_parquet(args.source / "synthetic_experts" / "expert_predictions.parquet")
    predictions = align_case_index(alerts, predictions)

    train = alerts.loc[alerts["month"].isin([3, 4, 5, 6])].copy()
    test = alerts.loc[alerts["month"].eq(7)].copy()
    train_predictions = predictions.loc[train.index]
    test_predictions = predictions.loc[test.index]

    edges = score_band_edges(train["model_score"], bands=10)
    global_skill, band_skill, model_correctness = historical_skill_tables(
        train, train_predictions, edges
    )
    train_bands = apply_score_bands(train["model_score"], edges)
    test["score_band"] = apply_score_bands(test["model_score"], edges)

    global_table = pd.DataFrame(
        {expert: [value] * len(band_skill) for expert, value in global_skill.items()},
        index=band_skill.index,
    )

    scenario_root = args.source / "testbed" / "test"
    records: list[dict[str, object]] = []
    assignment_records: list[pd.DataFrame] = []

    model_assignments = pd.Series("model", index=test.index, dtype="object")
    model_metrics = evaluate_assignments(test, test_predictions, model_assignments)
    records.append({"scenario": "all", "strategy": "model_only", "unused_capacity": 0, **model_metrics})

    for scenario in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
        capacity = pd.read_csv(scenario / "capacity.csv")

        random_result = random_assignment(test.index, capacity, random_state=42)
        global_result = greedy_advantage_assignment(
            test, capacity, global_table, model_correctness
        )
        specialist_result = greedy_advantage_assignment(
            test, capacity, band_skill, model_correctness
        )

        for strategy, result in (
            ("random_capacity", random_result),
            ("global_skill", global_result),
            ("risk_band_specialist", specialist_result),
        ):
            metrics = evaluate_assignments(test, test_predictions, result.assignments)
            records.append(
                {
                    "scenario": scenario.name,
                    "strategy": strategy,
                    "unused_capacity": result.unused_capacity,
                    **metrics,
                }
            )
            assignment_records.append(
                pd.DataFrame(
                    {
                        "case_id": test.index,
                        "scenario": scenario.name,
                        "strategy": strategy,
                        "assignment": result.assignments.to_numpy(),
                    }
                )
            )

    results = pd.DataFrame(records)
    summary = {
        "data_split": {
            "historical_alerts": len(train),
            "test_alerts": len(test),
            "historical_months": sorted(train["month"].unique().tolist()),
            "test_month": 7,
            "score_bands": int(train_bands.nunique()),
        },
        "scenario_count": int(results.loc[results["strategy"].ne("model_only"), "scenario"].nunique()),
        "model_only": results.loc[results["strategy"].eq("model_only")].iloc[0].to_dict(),
        "strategy_summary": (
            results.loc[results["strategy"].ne("model_only")]
            .groupby("strategy")
            .agg(
                scenarios=("scenario", "nunique"),
                mean_accuracy=("accuracy", "mean"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                mean_false_positive=("false_positive", "mean"),
                mean_false_negative=("false_negative", "mean"),
                mean_human_reviews=("reviewed_by_humans", "mean"),
                mean_unused_capacity=("unused_capacity", "mean"),
                accuracy_std=("accuracy", "std"),
                precision_std=("precision", "std"),
                recall_std=("recall", "std"),
                min_accuracy=("accuracy", "min"),
                max_accuracy=("accuracy", "max"),
            )
            .reset_index()
            .to_dict(orient="records")
        ),
        "all_scenarios": results.to_dict(orient="records"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.assignments.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.concat(assignment_records, ignore_index=True).to_parquet(args.assignments, index=False)
    print(json.dumps({key: value for key, value in summary.items() if key != "all_scenarios"}, indent=2))


if __name__ == "__main__":
    main()
