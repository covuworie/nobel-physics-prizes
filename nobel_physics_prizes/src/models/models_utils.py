def baseline_model_predict(features):
    """Perform a baseline model prediction.

    This model predicts that a physicist is a Nobel Laureate if the
    features `ratio_num_alma_mater` > 0.8 or `ratio_num_workplaces` > 0.
    Otherwise the physicist is predicted as a non-Laureate.  

    Args:
        features (pandas.Dataframe): Features Dataframe.

    Returns:
        pandas.Series: Target predictions.
            1 = Laureate, 0 = Non-laureate.
    """

    ratio_num_alma_mater = features.ratio_num_alma_mater > 0.8
    ratio_num_workplaces = features.ratio_num_workplaces > 0
    baseline = (ratio_num_alma_mater | ratio_num_workplaces).map(
        {True: 1, False: 0})
    return baseline
