import pandas as pd
import pytest

from src.models.models_utils import baseline_model_predict


@pytest.fixture
def expected_baseline_predict():
    expected = pd.Series(data=[
        1,
        1,
        0
    ])
    return expected


def test_baseline_model_predict(expected_baseline_predict):
    features = pd.DataFrame(data=[
        [1.2, 0.0, 1],
        [0.5, 0.9, 0],
        [0.8, 0.0, 0]],
        columns=['ratio_num_alma_mater', 'ratio_num_workplaces', 'born_in_USA'])
    baseline = baseline_model_predict(features)
    assert(baseline.equals(expected_baseline_predict))
