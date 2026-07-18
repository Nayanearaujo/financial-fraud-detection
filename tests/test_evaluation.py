import numpy as np
import pandas as pd

from fraud_detection.evaluation import precision_recall_at_capacity


def test_precision_recall_at_capacity() -> None:
    labels = pd.Series([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = precision_recall_at_capacity(labels, scores, capacity=2)
    assert result["fraud_captured"] == 2
    assert result["precision_at_capacity"] == 1.0
    assert result["recall_at_capacity"] == 1.0
