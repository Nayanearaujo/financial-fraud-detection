import pandas as pd

from fraud_detection.modelling import feature_columns, temporal_split


def test_feature_columns_supports_modern_string_dtype() -> None:
    data = pd.DataFrame(
        {
            "fraud_bool": [0, 1],
            "month": [0, 1],
            "income": [0.1, 0.2],
            "customer_age": [20, 30],
            "employment_status": ["CA", "CB"],
            "housing_status": ["BA", "BB"],
            "payment_type": pd.Series(["AA", "AB"], dtype="str"),
            "numeric_feature": [1.0, 2.0],
        }
    )
    numeric, categorical = feature_columns(data)
    assert numeric == ["numeric_feature"]
    assert categorical == ["payment_type"]


def test_temporal_split_excludes_incomplete_month() -> None:
    data = pd.DataFrame({"month": list(range(8)), "fraud_bool": [0] * 8})
    split = temporal_split(data)
    assert split.train["month"].tolist() == [0, 1, 2, 3]
    assert split.validation["month"].tolist() == [5, 6]
    assert split.test["month"].tolist() == [7]
