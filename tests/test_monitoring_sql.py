from pathlib import Path

import duckdb


def test_monitoring_views_build_from_published_aggregates() -> None:
    root = Path(__file__).resolve().parents[1]
    connection = duckdb.connect()
    connection.execute((root / "sql" / "monitoring_views.sql").read_text(encoding="utf-8"))

    assert connection.sql("SELECT COUNT(*) FROM monthly_fraud_monitoring").fetchone()[0] == 7
    assert connection.sql("SELECT COUNT(*) FROM model_capacity_comparison").fetchone()[0] == 8
    assert connection.sql("SELECT COUNT(*) FROM review_strategy_monitoring").fetchone()[0] == 3
