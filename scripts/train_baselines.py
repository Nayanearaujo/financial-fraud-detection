"""Train temporal fraud baselines and save validation/test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_detection.data_io import load_base
from fraud_detection.evaluation import classification_metrics, precision_recall_at_capacity
from fraud_detection.modelling import (
    build_histogram_pipeline,
    build_logistic_pipeline,
    split_xy,
    temporal_split,
    threshold_for_fbeta,
)
from fraud_detection.validation import remove_incomplete_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/model_metrics.json"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    return parser.parse_args()


def capacity_table(y_true: pd.Series, scores: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {
            "review_share": share,
            **precision_recall_at_capacity(y_true, scores, max(1, int(len(y_true) * share))),
        }
        for share in (0.01, 0.03, 0.05, 0.10)
    ]


def main() -> None:
    args = parse_args()
    base = remove_incomplete_rows(load_base(args.base))
    splits = temporal_split(base)
    x_train, y_train = split_xy(splits.train)
    x_validation, y_validation = split_xy(splits.validation)
    x_test, y_test = split_xy(splits.test)

    builders = {
        "logistic_regression": build_logistic_pipeline,
        "hist_gradient_boosting": build_histogram_pipeline,
    }
    results: dict[str, object] = {
        "split": {
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
            "excluded_month": 4,
        },
        "models": {},
    }
    args.models.mkdir(parents=True, exist_ok=True)

    for name, builder in builders.items():
        model = builder(splits.train)
        model.fit(x_train, y_train)
        validation_scores = model.predict_proba(x_validation)[:, 1]
        threshold = threshold_for_fbeta(y_validation, validation_scores, beta=2.0)
        test_scores = model.predict_proba(x_test)[:, 1]
        results["models"][name] = {
            "validation_threshold_f2": threshold,
            "validation": classification_metrics(y_validation, validation_scores, threshold),
            "validation_capacity": capacity_table(y_validation, validation_scores),
            "test": classification_metrics(y_test, test_scores, threshold),
            "test_capacity": capacity_table(y_test, test_scores),
        }
        joblib.dump(model, args.models / f"{name}.joblib")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
