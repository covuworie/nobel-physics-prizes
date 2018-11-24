import pandas as pd


def convert_categoricals_to_numerical(features):
    """Convert categorical features to numerical features.

    Args:
        features (pandas.DataFrame): Features dataframe.

    Returns:
        pandas.DataFrame: Features dataframe with categorical
            values ['yes', 'no'] and ['male', 'female']
            replaced with [1, 0].
    """

    features_numerical = features.set_index('full_name', drop=True)
    features_numerical = features_numerical.replace(
        to_replace={'yes': 1, 'no': 0, 'male': 1, 'female': 0})
    return features_numerical


def convert_target_to_numerical(target):
    """Convert target to numerical.

    Args:
        target (pandas.Series): Target.

    Returns:
        pandas.Series: Target with categorical ['yes', 'no'] replaced
            with [1, 0].
    """

    target_numerical = target.map({'yes': 1, 'no': 0})
    return target_numerical


def rank_hot_encode(features, encoder, columns=None):
    """Rank-hot encode ordinal features.

    Rank-hot encode ordinal features in the spirit of:
        http://scottclowe.com/2016-03-05-rank-hot-encoder/
    This code is drastically adapted from the code at which no longer works:
        https://stackoverflow.com/questions/50329109/rank-hot-encoding-python3

    Args:
        features (pandas.DataFrame): Features dataframe.
        encoder (sklearn.preprocessing.OneHotEncoder): One hot encoder. The fit
            method should have been previously called on the encoder.
        columns (list or pandas.Series, optional): Defaults to None. List of
            columns to encode.

    Returns:
        pandas.DataFrame: Features dataframe with rank-hot encoded values.
            The original columns are dropped.
    """

    if not columns:
        columns = features.columns

    encoded_columns = [
        '{}_at_least_{}'.format(col_name, val)
        for col_name, cat in zip(features[columns], encoder.categories_)
        for val in cat
    ]

    one_hot = encoder.transform(features[columns])
    rank_hot = one_hot.copy()

    col_start = 0
    cols_to_drop = []
    for _, cat in enumerate(encoder.categories_):
        cols_to_drop.append(col_start)
        col_end = col_start + len(cat)
        pos_first_zeros = (one_hot[:, col_start:col_end] != 0).argmax(axis=1)
        for row, pos_first_one in enumerate(pos_first_zeros):
            if pos_first_one == 0:
                continue
            # set all values before the first one to a one
            rank_hot[row, col_start:col_start + pos_first_one] = 1
        col_start = col_end

    enc_features = pd.DataFrame(
        rank_hot, index=features.index, columns=encoded_columns)
    # drop the first columns as they add no information (a value of exactly zero will
    # have all columns equal to zero)
    enc_features = enc_features.drop(
        enc_features.columns[cols_to_drop], axis='columns')

    features_with_rank_hot = features.drop(columns, axis='columns')
    features_with_rank_hot = features_with_rank_hot.join(enc_features)
    return features_with_rank_hot
