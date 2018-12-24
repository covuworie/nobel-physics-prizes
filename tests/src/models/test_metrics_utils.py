import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, confusion_matrix, matthews_corrcoef, roc_curve
from sklearn.preprocessing import binarize

from src.models.metrics_utils import (confusion_matrix_to_dataframe,
                                      mcc_auc_score, mcc_curve)


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


@pytest.fixture
def y_true_y_score():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    return y_true, y_score


@pytest.fixture
def expected_roc_curve(y_true_y_score):
    y_true, y_score = y_true_y_score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return fpr, tpr, thresholds


@pytest.fixture
def expected_roc_curve_probability(expected_roc_curve):
    fpr, tpr, thresholds = expected_roc_curve
    thresholds[0] = 1.0
    thresholds = np.append(thresholds, 0.0)
    fpr = np.append(fpr, 1.0)
    tpr = np.append(tpr, 1.0)
    return fpr, tpr, thresholds


@pytest.fixture
def expected_mcc_curve(y_true_y_score, expected_roc_curve):
    y_true, y_score = y_true_y_score
    _, _, thresholds = expected_roc_curve
    mccs = []
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype('int64')
        mcc = matthews_corrcoef(y_true, y_pred)
        mccs.append(mcc)
    return np.array(mccs), thresholds


@pytest.fixture
def expected_mcc_curve_probability(y_true_y_score, expected_roc_curve_probability):
    y_true, y_score = y_true_y_score
    _, _, thresholds = expected_roc_curve_probability
    mccs = []
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype('int64')
        mcc = matthews_corrcoef(y_true, y_pred)
        mccs.append(mcc)
    return np.array(mccs), thresholds


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


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_mcc_curve(y_true_y_score, expected_mcc_curve):
    y_true, y_score = y_true_y_score
    expected_mccs, expected_thresholds = expected_mcc_curve
    mccs, thresholds = mcc_curve(y_true, y_score, probability=False)
    np.testing.assert_allclose(thresholds, expected_thresholds)
    np.testing.assert_allclose(mccs, expected_mccs)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_mcc_curve_probability(
        y_true_y_score, expected_mcc_curve_probability):
    y_true, y_score = y_true_y_score
    expected_mccs, expected_thresholds = expected_mcc_curve_probability
    mccs, thresholds = mcc_curve(y_true, y_score, probability=True)
    np.testing.assert_allclose(thresholds, expected_thresholds)
    np.testing.assert_allclose(mccs, expected_mccs)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_mcc_auc_score(y_true_y_score, expected_mcc_curve):
    y_true, y_score = y_true_y_score
    mccs, thresholds = expected_mcc_curve

    expected_mcc_auc = auc(thresholds, mccs)
    mcc_auc = mcc_auc_score(
        y_true, y_score, probability=False, normalize=False)
    np.testing.assert_allclose(mcc_auc, expected_mcc_auc)
    mcc_auc = mcc_auc_score(y_true, y_score, probability=False, normalize=True)

    normalized_thresholds = (
        (thresholds - np.min(thresholds)) / (np.max(thresholds) - np.min(thresholds)))
    expected_mcc_auc = auc(normalized_thresholds, mccs)
    np.testing.assert_allclose(mcc_auc, expected_mcc_auc)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_mcc_auc_score_probability(y_true_y_score, expected_mcc_curve_probability):
    y_true, y_score = y_true_y_score
    mccs, thresholds = expected_mcc_curve_probability

    expected_mcc_auc = auc(thresholds, mccs)
    mcc_auc = mcc_auc_score(y_true, y_score, probability=True, normalize=False)
    np.testing.assert_allclose(mcc_auc, expected_mcc_auc)

    mcc_auc = mcc_auc_score(y_true, y_score, probability=True, normalize=True)
    np.testing.assert_allclose(mcc_auc, expected_mcc_auc)
