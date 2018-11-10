import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.models.metrics_utils import confusion_matrix_to_dataframe


@pytest.fixture
def expected_confusion_matrix_numpy():
    expected = np.array([[0, 2, 2],
                         [1, 1, 2],
                         [1, 3, 4]], dtype='int64')
    return expected


@pytest.fixture
def expected_confusion_matrix_default(expected_confusion_matrix_numpy):
    expected = pd.DataFrame(
        data=expected_confusion_matrix_numpy,
        index=['Observed negative', 'Observed positive', 'Predicted total'],
        columns=['Predicted negative', 'Predicted positive', 'Observed total'])
    return expected


@pytest.fixture
def expected_confusion_matrix(expected_confusion_matrix_numpy):
    expected = pd.DataFrame(
        data=expected_confusion_matrix_numpy,
        index=['Measured negative', 'Measured positive', 'Classified total'],
        columns=['Classified negative', 'Classified positive', 'Measured total'])
    return expected


def test_confusion_matrix_to_dataframe_default_values(
        expected_confusion_matrix_default):
    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_df = confusion_matrix_to_dataframe(conf_matrix)
    assert(conf_matrix_df.equals(expected_confusion_matrix_default))


def test_confusion_matrix_to_dataframe(
        expected_confusion_matrix):
    y_true = [0, 1, 0, 1]
    y_pred = [1, 1, 1, 0]
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_df = confusion_matrix_to_dataframe(
        conf_matrix, index=expected_confusion_matrix.index[0:2],
        columns=expected_confusion_matrix.columns[0:2],
        index_total_label='Measured total',
        column_total_label='Classified total')
    assert(conf_matrix_df.equals(expected_confusion_matrix))
