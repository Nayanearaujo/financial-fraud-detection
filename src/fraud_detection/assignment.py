"""Capacity-aware assignment policies for the synthetic fraud-review team."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssignmentResult:
    assignments: pd.Series
    reviewed_cases: int
    unused_capacity: int


def score_band_edges(scores: pd.Series, bands: int = 10) -> np.ndarray:
    """Fit stable quantile boundaries on historical model scores."""
    _, edges = pd.qcut(scores, q=bands, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_score_bands(scores: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = list(range(len(edges) - 1))
    return pd.cut(scores, bins=edges, labels=labels, include_lowest=True).astype(int)


def historical_skill_tables(
    alerts: pd.DataFrame,
    expert_predictions: pd.DataFrame,
    edges: np.ndarray,
    smoothing: float = 20.0,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Estimate global and risk-band correctness using historical alerts only."""
    if not alerts.index.equals(expert_predictions.index):
        raise ValueError("Alert and expert-prediction indexes must align.")

    correctness = expert_predictions.eq(alerts["fraud_bool"], axis=0)
    global_skill = correctness.mean(axis=0)
    bands = apply_score_bands(alerts["model_score"], edges)

    records: list[dict[str, object]] = []
    for band in sorted(bands.unique()):
        in_band = bands.eq(band)
        counts = int(in_band.sum())
        correct = correctness.loc[in_band].sum(axis=0)
        smoothed = (correct + smoothing * global_skill) / (counts + smoothing)
        records.extend(
            {"band": int(band), "expert": expert, "expected_correctness": float(value)}
            for expert, value in smoothed.items()
        )
    band_skill = pd.DataFrame(records).pivot(index="band", columns="expert", values="expected_correctness")

    # The screening model marks every supplied alert as positive. Its historical
    # correctness is therefore the fraud rate within each score band.
    model_correctness = alerts.groupby(bands, observed=True)["fraud_bool"].mean()
    return global_skill, band_skill, model_correctness


def _normalise_capacity(capacity: pd.DataFrame) -> tuple[dict[str, int], int]:
    ignored = {"batch_id", "batch_size", "Unnamed: 0"}
    row = capacity.iloc[0]
    available = {
        column: int(row[column])
        for column in capacity.columns
        if column not in ignored and int(row[column]) > 0
    }
    return available, sum(available.values())


def random_assignment(
    case_ids: pd.Index,
    capacity: pd.DataFrame,
    random_state: int = 42,
) -> AssignmentResult:
    """Assign a random case order while respecting each analyst's maximum."""
    available, total_capacity = _normalise_capacity(capacity)
    rng = np.random.default_rng(random_state)
    shuffled_cases = rng.permutation(case_ids.to_numpy())
    expert_slots = np.concatenate(
        [np.repeat(expert, slots) for expert, slots in available.items()]
    )
    rng.shuffle(expert_slots)
    reviewed = min(len(shuffled_cases), len(expert_slots))
    result = pd.Series(index=case_ids, data="model", dtype="object")
    result.loc[shuffled_cases[:reviewed]] = expert_slots[:reviewed]
    return AssignmentResult(result, reviewed, total_capacity - reviewed)


def greedy_advantage_assignment(
    alerts: pd.DataFrame,
    capacity: pd.DataFrame,
    expected_expert_correctness: pd.DataFrame,
    expected_model_correctness: pd.Series,
) -> AssignmentResult:
    """Assign analyst reviews only where expected correctness exceeds the model."""
    available, total_capacity = _normalise_capacity(capacity)
    experts = [expert for expert in available if expert in expected_expert_correctness.columns]
    if not experts:
        raise ValueError("No active experts have historical skill estimates.")

    candidates: list[tuple[float, int, str]] = []
    for case_id, band in alerts["score_band"].items():
        model_probability = float(expected_model_correctness.loc[band])
        for expert in experts:
            advantage = float(expected_expert_correctness.loc[band, expert]) - model_probability
            if advantage > 0:
                candidates.append((advantage, int(case_id), expert))
    candidates.sort(reverse=True)

    assignments = pd.Series(index=alerts.index, data="model", dtype="object")
    remaining = available.copy()
    reviewed = 0
    for _, case_id, expert in candidates:
        if assignments.loc[case_id] != "model" or remaining[expert] <= 0:
            continue
        assignments.loc[case_id] = expert
        remaining[expert] -= 1
        reviewed += 1
        if reviewed == total_capacity:
            break
    return AssignmentResult(assignments, reviewed, total_capacity - reviewed)


def evaluate_assignments(
    alerts: pd.DataFrame,
    expert_predictions: pd.DataFrame,
    assignments: pd.Series,
) -> dict[str, float | int]:
    """Evaluate final model-or-human decisions for the alert population."""
    if not alerts.index.equals(expert_predictions.index) or not alerts.index.equals(assignments.index):
        raise ValueError("Alerts, predictions and assignments must share the same index.")

    truth = alerts["fraud_bool"].astype(int)
    # All records in this table crossed the screening threshold, so the model's
    # alert decision is positive unless the case is deferred to an analyst.
    decisions = pd.Series(1, index=alerts.index, dtype=int)
    human_mask = assignments.ne("model")
    for expert in assignments.loc[human_mask].unique():
        cases = assignments.eq(expert)
        decisions.loc[cases] = expert_predictions.loc[cases, expert].astype(int)

    true_positive = int(((decisions == 1) & (truth == 1)).sum())
    false_positive = int(((decisions == 1) & (truth == 0)).sum())
    false_negative = int(((decisions == 0) & (truth == 1)).sum())
    true_negative = int(((decisions == 0) & (truth == 0)).sum())
    return {
        "alerts": len(alerts),
        "reviewed_by_humans": int(human_mask.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "accuracy": float((true_positive + true_negative) / len(alerts)),
        "precision": float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0,
        "recall": float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0,
        "positive_decisions": int((decisions == 1).sum()),
    }
