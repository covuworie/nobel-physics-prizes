def convert_categoricals_to_numerical(features):
    """Convert categorical features to numerical features.

    Args:
        features (pandas.Dataframe): Features dataframe.

    Returns:
        pandas.Dataframe: Features dataframe with categorical
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
