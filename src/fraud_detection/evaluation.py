"""Metrics suited to rare-event fraud detection and investigation queues."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "alert_rate": float(predictions.mean()),
    }


def precision_recall_at_capacity(
    y_true: pd.Series,
    scores: np.ndarray,
    capacity: int,
) -> dict[str, float | int]:
    """Evaluate the highest-risk cases under a fixed review capacity."""
    if capacity <= 0 or capacity > len(y_true):
        raise ValueError("capacity must be between 1 and the number of cases")
    order = np.argsort(-np.asarray(scores))[:capacity]
    selected = np.asarray(y_true)[order]
    captured = int(selected.sum())
    total_fraud = int(np.asarray(y_true).sum())
    return {
        "review_capacity": int(capacity),
        "fraud_captured": captured,
        "precision_at_capacity": float(captured / capacity),
        "recall_at_capacity": float(captured / total_fraud) if total_fraud else 0.0,
    }
