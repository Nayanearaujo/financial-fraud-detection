"""Run portable DuckDB checks over the dashboard's publishable aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql", type=Path, default=Path("sql/monitoring_views.sql"))
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring_checks.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection = duckdb.connect()
    connection.execute(args.sql.read_text(encoding="utf-8"))

    checks = {
        "complete_months": connection.sql(
            "SELECT COUNT(*) FROM monthly_fraud_monitoring"
        ).fetchone()[0],
        "test_capacity_levels": connection.sql(
            "SELECT COUNT(*) FROM model_capacity_comparison WHERE split = 'test'"
        ).fetchone()[0],
        "assignment_policies": connection.sql(
            "SELECT COUNT(*) FROM review_strategy_monitoring"
        ).fetchone()[0],
        "month_four_excluded": connection.sql(
            "SELECT COUNT(*) = 0 FROM monthly_fraud_monitoring WHERE source_month = 4"
        ).fetchone()[0],
    }

    expected = {
        "complete_months": 7,
        "test_capacity_levels": 4,
        "assignment_policies": 3,
        "month_four_excluded": True,
    }
    if checks != expected:
        raise ValueError(f"Monitoring checks failed: {checks}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
