"""Quality checks for the supplied FiFAR and BAF files."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    duplicate_rows: int
    missing_cells: int
    incomplete_rows: int
    fraud_cases: int
    fraud_rate: float
    months: list[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_base(data: pd.DataFrame) -> DataQualityReport:
    """Summarise completeness and target distribution without imputation."""
    required = {"fraud_bool", "month"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    incomplete = data.isna().any(axis=1)
    valid_target = data["fraud_bool"].dropna()
    months = sorted(data["month"].dropna().astype(int).unique().tolist())
    return DataQualityReport(
        rows=len(data),
        columns=len(data.columns),
        duplicate_rows=int(data.duplicated().sum()),
        missing_cells=int(data.isna().sum().sum()),
        incomplete_rows=int(incomplete.sum()),
        fraud_cases=int(valid_target.sum()),
        fraud_rate=float(valid_target.mean()),
        months=months,
    )


def remove_incomplete_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Remove only structurally incomplete rows and preserve source sentinels."""
    return data.loc[~data.isna().any(axis=1)].copy()


def monthly_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return application volume, fraud count and fraud rate by source month."""
    return (
        data.groupby("month", observed=True)["fraud_bool"]
        .agg(applications="size", fraud_cases="sum", fraud_rate="mean")
        .reset_index()
        .astype({"month": int, "applications": int, "fraud_cases": int})
    )
