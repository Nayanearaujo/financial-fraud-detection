import pandas as pd

from fraud_detection.validation import audit_base, monthly_summary, remove_incomplete_rows


def test_audit_and_remove_incomplete_rows() -> None:
    data = pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0],
            "month": [0, 1, None],
            "feature": [1.0, 2.0, None],
        }
    )
    report = audit_base(data)
    assert report.rows == 3
    assert report.incomplete_rows == 1
    assert report.fraud_cases == 1
    assert len(remove_incomplete_rows(data)) == 2


def test_monthly_summary() -> None:
    data = pd.DataFrame({"fraud_bool": [0, 1, 1], "month": [0, 0, 1]})
    result = monthly_summary(data)
    assert result.loc[result["month"] == 0, "applications"].item() == 2
    assert result.loc[result["month"] == 1, "fraud_rate"].item() == 1.0
