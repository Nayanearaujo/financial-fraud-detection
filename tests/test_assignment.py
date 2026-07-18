import numpy as np
import pandas as pd

from fraud_detection.assignment import (
    apply_score_bands,
    evaluate_assignments,
    random_assignment,
    score_band_edges,
)


def test_score_bands_cover_new_scores() -> None:
    edges = score_band_edges(pd.Series(np.linspace(0.05, 0.95, 100)), bands=5)
    result = apply_score_bands(pd.Series([0.01, 0.5, 0.99]), edges)
    assert result.notna().all()
    assert result.min() == 0
    assert result.max() == 4


def test_random_assignment_respects_capacity() -> None:
    capacity = pd.DataFrame(
        {"batch_id": [1], "batch_size": [5], "standard#0": [2], "standard#1": [1]}
    )
    result = random_assignment(pd.Index([10, 11, 12, 13, 14]), capacity)
    assert result.reviewed_cases == 3
    assert result.assignments.eq("standard#0").sum() == 2
    assert result.assignments.eq("standard#1").sum() == 1


def test_evaluate_assignments_uses_model_for_unreviewed_cases() -> None:
    index = pd.Index([10, 11, 12])
    alerts = pd.DataFrame({"fraud_bool": [0, 1, 0]}, index=index)
    predictions = pd.DataFrame({"standard#0": [0, 0, 1]}, index=index)
    assignments = pd.Series(["standard#0", "model", "model"], index=index)
    metrics = evaluate_assignments(alerts, predictions, assignments)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 0
