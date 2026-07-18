import pandas as pd
import pytest

from fraud_detection.data_io import align_case_index


def test_align_case_index_reorders_by_identifier() -> None:
    reference = pd.DataFrame({"value": [1, 2]}, index=[10, 20])
    records = pd.DataFrame({"decision": [0, 1]}, index=[20, 10])

    aligned = align_case_index(reference, records)

    assert aligned.index.tolist() == [10, 20]
    assert aligned["decision"].tolist() == [1, 0]


def test_align_case_index_rejects_different_case_sets() -> None:
    reference = pd.DataFrame(index=[10, 20])
    records = pd.DataFrame(index=[10, 30])

    with pytest.raises(ValueError, match="identifier sets"):
        align_case_index(reference, records)
