import pandas as pd


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
