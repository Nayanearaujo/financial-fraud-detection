"""Temporal model training for the application-level fraud task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from fraud_detection.features import AUDIT_ONLY_FEATURES, TARGET, TIME_COLUMN


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(data: pd.DataFrame) -> TemporalSplit:
    """Apply the documented split and keep month 4 out of primary modelling."""
    return TemporalSplit(
        train=data.loc[data[TIME_COLUMN].isin([0, 1, 2, 3])].copy(),
        validation=data.loc[data[TIME_COLUMN].isin([5, 6])].copy(),
        test=data.loc[data[TIME_COLUMN].eq(7)].copy(),
    )


def feature_columns(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {TARGET, TIME_COLUMN, *AUDIT_ONLY_FEATURES}
    features = [column for column in data.columns if column not in excluded]
    categorical = [
        column
        for column in features
        if isinstance(data[column].dtype, pd.CategoricalDtype)
        or is_object_dtype(data[column])
        or is_string_dtype(data[column])
    ]
    numeric = [column for column in features if column not in categorical]
    return numeric, categorical


def build_logistic_pipeline(data: pd.DataFrame) -> Pipeline:
    numeric, categorical = feature_columns(data)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def build_histogram_pipeline(data: pd.DataFrame) -> Pipeline:
    numeric, categorical = feature_columns(data)
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("prepare", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    learning_rate=0.08,
                    max_iter=150,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def split_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=[TARGET]), data[TARGET].astype("int8")


def threshold_for_fbeta(y_true: pd.Series, scores: np.ndarray, beta: float = 2.0) -> float:
    """Select a validation threshold that gives recall additional weight."""
    candidates = np.quantile(scores, np.linspace(0.80, 0.999, 300))
    truth = np.asarray(y_true)
    beta_squared = beta**2
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = np.logical_and(predicted, truth == 1).sum()
        false_positive = np.logical_and(predicted, truth == 0).sum()
        false_negative = np.logical_and(~predicted, truth == 1).sum()
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        denominator = beta_squared * precision + recall
        score = (1 + beta_squared) * precision * recall / denominator if denominator else 0.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold
