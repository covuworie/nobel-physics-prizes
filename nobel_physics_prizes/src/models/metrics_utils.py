import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import auc, matthews_corrcoef, roc_curve
from sklearn.preprocessing import binarize


def print_matthews_corrcoef(corrcoef, classifier_name, data_label='train'):
    """Print Matthews Correlation Coefficient.

    Args:
        corrcoef (float): Matthews Correlation Coefficient.
        classifier_name (str): Name of classfier.
        data_label (str, optional): Defaults to 'train'. Data label.
    """

    print(classifier_name + ' MCC ({0}): {1}'.format(
        data_label, round(corrcoef, 2)))


def confusion_matrix_to_dataframe(
        confusion_matrix, index=['Observed negative', 'Observed positive'],
        columns=['Predicted negative', 'Predicted positive'],
        index_total_label='Observed total',
        column_total_label='Predicted total'):
    """Convert a confusion matrix into a pandas dataframe.

    Convert a confusion matrix to a pandas dataframe adding nice row and column labels
    and also computing the row and column totals.

    Args:
        confusion_matrix (np.array, shape = [n_classes, n_classes]): sklearn
            confusion matrix.  
        index (list, optional): Defaults to ['Observed negative', 'Observed positive'].
            Dataframe index.
        columns (list, optional): Defaults to ['Predicted negative', 'Predicted positive'].
            Dataframe columns.
        index_total_label (str, optional): Defaults to 'Observed total'. Label for the
            index totals.
        column_total_label (str, optional): Defaults to 'Predicted total'. Label for the
            column totals.

    Returns:
        pd.Dataframe: Confusion matrix with labels and row and column totals.
    """

    confusion_matrix_df = pd.DataFrame(
        data=confusion_matrix, columns=columns, index=index)

    observed_total = confusion_matrix_df.sum(axis='columns')
    observed_total.name = index_total_label
    predicted_total = confusion_matrix_df.sum(axis='rows')
    predicted_total.name = column_total_label

    confusion_matrix_df = confusion_matrix_df.append(predicted_total)
    confusion_matrix_df = confusion_matrix_df.join(observed_total)
    confusion_matrix_df.loc[column_total_label, index_total_label] = (
        confusion_matrix_df.sum()[index_total_label])
    return confusion_matrix_df.astype('int64')


def mcc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True, probability=True):
    """Compute Matthews correlation coefficient (MCC) curve.

    Note: this implementation is restricted to the binary classification task.

    Args:
        y_true (array, shape = [n_samples]): True binary labels. If labels are not either {-1, 1} or {0, 1},
            then pos_label should be explicitly given.
        y_score (array, shape = [n_samples]): Target scores, can either be probability estimates of the
            positive class, confidence values, or non-thresholded measure of decisions (as returned by 
            `decision_function` on some classifiers).
        pos_label (int or str, optional): Defaults to None. Label considered as positive and others are
            considered negative.
        sample_weight (array-like of shape = [n_samples], optional): Defaults to None. Sample weights.
        drop_intermediate (bool, optional): Defaults to True. Whether to drop some suboptimal thresholds
            which would not appear on a plotted MCC curve. This is useful in order to create lighter MCC curves.
        probability (bool, optional): Defaults to True. Whether `y_score` are probability estimates of the
            positive class. If True then the thresholds are bounded in [0, 1]. The `sklearn` learn value of
            thresholds[0] from `roc_curve` is replaced by 1.0 and a value is appended to the end of thresholds
            such that thresholds[-1] is equal to 0 (and the false positive and true positive rates are set to 1.0). 

    Returns:
        mccs : array, shape = [>2]
            Matthews Correlation Coefficients such that element i is the MCC of predictions with
            score >= thresholds[i].

        thresholds : array, shape = [n_thresholds]
            Decreasing thresholds on the decision function used to compute MCC.
    """

    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=pos_label,
                                     sample_weight=sample_weight, drop_intermediate=drop_intermediate)

    # If y_score is a probability then bound the thresholds between 0 and 1. Set the first (highest)
    # value to 1 and append a last (lowest) value equal to 0.
    # See: https://github.com/scikit-learn/scikit-learn/issues/3097
    if probability:
        thresholds[0] = 1.0
        thresholds = np.append(thresholds, 0.0)
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)

    mccs = []
    with warnings.catch_warnings():  # ignore runtime warnings caused by zero MCC
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for threshold in thresholds:
            y_pred = binarize(y_score.reshape(-1, 1),
                              threshold=threshold).astype('int64')
            mcc = matthews_corrcoef(
                y_true, y_pred, sample_weight=sample_weight)
            mccs.append(mcc)

    return np.array(mccs), thresholds


def mcc_auc_score(y_true, y_score, sample_weight=None, probability=True, normalize=True):
    """Compute Area Under the Matthews Correlation Coefficient Curve (MCC AUC) from prediction scores.

    Note: this implementation is restricted to the binary classification task or multilabel classification 
    task in label indicator format.

    Args:
        y_true (array, shape = [n_samples] or [n_samples, n_classes]): True binary labels or binary label
            indicators.
        y_score (array, shape = [n_samples] or [n_samples, n_classes]): Target scores, can either be
            probability estimates of the positive class, confidence values, or non-thresholded measure of
            decisions (as returned by `decision_function` on some classifiers). For binary y_true, y_score
            is supposed to be the score of the class with greater label.
        sample_weight (array-like of shape = [n_samples], optional): Defaults to None. Sample weights.
        probability (bool, optional): Defaults to True. Whether `y_score` are probability estimates of the
            positive class. If True then the thresholds are bounded in [0, 1]. The `sklearn` learn value of
            thresholds[0] from `roc_curve` is replaced by 1.0 and a value is appended to the end of thresholds
            such that thresholds[-1] is equal to 0 (and the false positive and true positive rates are set to 1.0). 
        normalize (bool, optional): Defaults to True. Whether to normalize the MCC AUC so that it bounded in
            [0, 1]. The normalization constant is the maximum possible AUC across the support of the thresholds
            (i.e. its minimum and maximum values). This is equivalent to assuming an MCC = 1.0 for all thresholds.
            Note that for `probability`=True setting `normalize`=True should have no effect as the normalization
            constant is equal to 1.0.

    Returns:
        auc: float
            Area under the curve.
    """

    mccs, thresholds = mcc_curve(
        y_true, y_score, sample_weight=sample_weight, probability=probability)
    mcc_auc = auc(thresholds, mccs)

    if not normalize:
        return mcc_auc

    mccs_ones = np.ones(2)
    mcc_ones_auc = auc(np.array([thresholds[0], thresholds[-1]]), mccs_ones)
    mcc_auc /= mcc_ones_auc
    if probability:
        assert(0.0 <= mcc_auc <= 1.0)

    return mcc_auc
