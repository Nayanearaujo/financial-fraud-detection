"""Data loading helpers with explicit schema and path checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_COLUMNS = [
    "fraud_bool",
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "payment_type",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "employment_status",
    "credit_risk_score",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "source",
    "session_length_in_minutes",
    "device_os",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
]


def require_file(path: str | Path) -> Path:
    """Return a resolved file path or raise a clear error."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required data file was not found: {resolved}")
    return resolved


def load_base(path: str | Path) -> pd.DataFrame:
    """Load the base application data without silently altering sentinels."""
    data = pd.read_csv(require_file(path))
    if list(data.columns) != BASE_COLUMNS:
        raise ValueError("Base.csv columns do not match the documented schema.")
    return data


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load an official FiFAR parquet table."""
    return pd.read_parquet(require_file(path))


def align_case_index(reference: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Return records ordered by a verified one-to-one case identifier index."""
    if not reference.index.is_unique or not records.index.is_unique:
        raise ValueError("Case identifiers must be unique before alignment.")
    if set(reference.index) != set(records.index):
        raise ValueError("Case identifier sets do not match.")
    return records.reindex(reference.index)
