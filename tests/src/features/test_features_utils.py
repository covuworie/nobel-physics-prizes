import pandas as pd
import pytest

from src.features.features_utils import (convert_categoricals_to_numerical,
                                         convert_target_to_numerical)


@pytest.fixture()
def expected_features():
    expected = pd.DataFrame(data=[
        [1.4, 1, 1],
        [2.0, 0, 1],
        [2.8, 1, 0]
    ],
        index=['Sheldon Cooper', 'Amy Fowler', 'Mandark'],
        columns=['ratio_num_workplaces', 'gender', 'worked_in_USA'])
    return expected


@pytest.fixture()
def expected_target():
    expected = pd.Series(data=[
        1,
        0,
        0
    ],
        name='physics_laureate')
    return expected


def test_convert_categoricals_to_numerical(expected_features):
    features = pd.DataFrame(data=[
        ['Sheldon Cooper', 1.4, 'male', 'yes'],
        ['Amy Fowler', 2.0, 'female', 'yes'],
        ['Mandark', 2.8, 'male', 'no']
    ],
        columns=['full_name', 'ratio_num_workplaces', 'gender', 'worked_in_USA'])
    features_numerical = convert_categoricals_to_numerical(features)
    assert(features_numerical.equals(expected_features))


def test_convert_target_to_numerical(expected_target):
    target = expected_target.map({1: 'yes', 0: 'no'})
    target_numerical = convert_target_to_numerical(target)
    assert(target_numerical.equals(expected_target))
