import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import indices_to_mask


def bootstrap_prediction(X, y, score_func, base_estimator=None, n_estimators=10,
                         max_samples=1.0, max_features=1.0, bootstrap=True,
                         bootstrap_features=False, n_jobs=None, random_state=None):
    """Bootstrap the scores from an `sklearn` estimator.

    Args:
        X (array-like, dtype=float64, , size=[n_samples, n_features]): Feature matrix.
        y (array, dtype=float64, size=[n_samples]): Target vector.
        score_func (callable): Score function (or loss function) with signature
            score_func(y, y_pred, **kwargs).
        base_estimator (object or None, optional): Defaults to None. The base estimator
            to fit on random subsets of the dataset. If None, then the base estimator
            is a decision tree.
        n_estimators (int, optional): Defaults to 10. The number of base estimators in
            the ensemble.
        max_samples (int or float, optional): Defaults to 1.0. The number of samples
            to draw from X to train each base estimator. If int, then draw max_samples
            samples. If float, then draw max_samples * X.shape[0] samples.
        max_features (int or float, optional): Defaults to 1.0. The number of features
            to draw from X to train each base estimator. If int, then draw max_features
            features. If float, then draw max_features * X.shape[1] features.
        bootstrap (bool, optional): Defaults to True. Whether samples are drawn with
            replacement.
        bootstrap_features (bool, optional): Defaults to False. Whether features are
            drawn with replacement.
        n_jobs (int or None, optional): Defaults to None. The number of jobs to run in
            parallel for both fit and predict. None means 1 unless in a
            joblib.parallel_backend context.
        random_state (int, RandomState instance or None, optional): Defaults to None.
            If int, random_state is the seed used by the random number generator; If
            RandomState instance, random_state is the random number generator; If None,
            the random number generator is the RandomState instance used by np.random.

    Returns:
        numpy.ndarray: Distribution of score function statistic.
    """

    bag = BaggingClassifier(
        base_estimator=base_estimator, n_estimators=n_estimators, max_samples=max_samples,
        max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features,
        n_jobs=n_jobs, random_state=random_state)
    bag.fit(X, y)

    stats = []
    for estimator, samples in zip(bag.estimators_, bag.estimators_samples_):
        # Create mask for OOB samples
        mask = ~indices_to_mask(samples, len(y))

        # Compute predictions on out-of-bag samples
        y_pred = estimator.predict(X[mask])

        # Compute statistic
        stat = score_func(y[mask], y_pred)
        stats.append(stat)

    stats = np.array(stats)
    return stats


def percentile_conf_int(data, alpha=0.05):
    """Compute the percentile confidence interval from some data.

    Args:
        data (numpy.ndarray): Data.
        alpha (float, optional): Defaults to 0.05. Significance level.
            (0 < alpha < 1). 

    Returns:
        np.ndarray: Lower and upper percentile confidence interval.
    """

    lower = 100 * alpha / 2
    upper = 100 * (1. - alpha / 2)
    conf_int = np.percentile(data, [lower, upper])
    return conf_int
