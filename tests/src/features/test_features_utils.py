import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.features.features_utils import (convert_categoricals_to_numerical,
                                         convert_target_to_numerical, rank_hot_encode)


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


@pytest.fixture()
def ordinal_features():
    ordinal = pd.DataFrame(data=[
        [1, 3, 0],
        [0, 1, 1],
        [2, 4, 2]
    ], index=[0, 1, 2], columns=['ord_1', 'ord_2', 'ord_3'])
    return ordinal


@pytest.fixture()
def binary_features():
    binary = pd.DataFrame(data=[
        [1],
        [0],
        [0]
    ], index=[0, 1, 2], columns=['bin_1'])
    return binary


@pytest.fixture()
def expected_rank_hot_encode():
    ordinal = pd.DataFrame(data=[
        [1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ], dtype='int64', index=[0, 1, 2], columns=[
        'ord_1_at_least_1', 'ord_1_at_least_2',
        'ord_2_at_least_3', 'ord_2_at_least_4',
        'ord_3_at_least_1', 'ord_3_at_least_2'])
    return ordinal


@pytest.fixture()
def expected_rank_hot_encode_columns(expected_rank_hot_encode, binary_features):
    expected = binary_features.join(expected_rank_hot_encode)
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


def test_rank_hot_encode_all_features(ordinal_features, expected_rank_hot_encode):
    enc = OneHotEncoder(categories='auto', sparse=False,
                        dtype='int64', handle_unknown='ignore')
    enc.fit(ordinal_features)
    rank_hot = rank_hot_encode(ordinal_features, enc)
    assert(rank_hot.equals(expected_rank_hot_encode))


def test_rank_hot_encode_columns(binary_features, ordinal_features,
                                 expected_rank_hot_encode_columns):
    enc = OneHotEncoder(categories='auto', sparse=False,
                        dtype='int64', handle_unknown='ignore')
    mixed_features = ordinal_features.join(binary_features)
    enc.fit(mixed_features[ordinal_features.columns])
    rank_hot = rank_hot_encode(
        mixed_features, enc, columns=ordinal_features.columns.tolist())
    assert(rank_hot.equals(expected_rank_hot_encode_columns))
