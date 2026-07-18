"""Feature policy for modelling and subgroup evaluation."""

from __future__ import annotations

import pandas as pd


TARGET = "fraud_bool"
TIME_COLUMN = "month"
CATEGORICAL_FEATURES = ["payment_type", "source", "device_os"]
AUDIT_ONLY_FEATURES = ["income", "customer_age", "employment_status", "housing_status"]


def modelling_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the primary feature matrix while reserving sensitive fields for audit."""
    excluded = {TARGET, TIME_COLUMN, *AUDIT_ONLY_FEATURES}
    features = data.drop(columns=[column for column in excluded if column in data])
    target = data[TARGET].astype("int8")
    return features, target


def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame:
    """Encode the non-sensitive categorical fields with stable column names."""
    categorical = [column for column in CATEGORICAL_FEATURES if column in data]
    return pd.get_dummies(data, columns=categorical, drop_first=False, dtype="int8")
