import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_bootstrap_statistics(stats, test_stat, ci, alpha, test_stat_label, x_label, title=None):
    """Plot a histogram of a bootstrap statistic.

    Args:
        stats (array): Statitics.
        test_stat (float): The value of the test statistic.
        ci (array, shape(2,)): Confidence interval. [lower ci, upper ci]
        alpha (float): Significance level. 0.0 < alpha < 1.0.
        test_stat_label (str): Test statistic label.
        x_label (str): x-axis label.
        title (str, optional): Defaults to None. Plot title.

    Returns:
        matplotlib.axes.Axes: axes.
    """

    _, ax = plt.subplots(figsize=(8, 6))
    ax.hist(stats)
    ax.axvline(x=test_stat, color='blue', linestyle='-',
               label=test_stat_label + str(round(test_stat, 3)))
    ci_label = ['lower ' + str(round(100 * (1 - alpha))) + '%',
                'upper ' + str(round(100 * (1 - alpha))) + '%']
    linestyles = ['--', '-.']
    for i, val in enumerate(ci):
        label = str(ci_label[i])
        linestyle = linestyles[i]
        ax.axvline(x=val, color='blue', linestyle=linestyle,
                   label=label + ' CI: ' + str(round(val, 3)))

    ax.set_ylabel('Frequency')
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    return ax


def plot_logistic_regression_odds_ratio(coefs, top_n=1, columns=None, title=None, plotting_context='notebook'):
    """Plot a bar chart of the odds ratio for features in a logistic regression model from `sklearn`.

    Args:
        coefs (array, shape (1, n_features)): Coefficient of the features in the decision function.
        top_n (`int` or `float`, optional): Defaults to 1. If `int` then plot the top number of features
            according to the odds ratio. If `float` then plot all features above this odds ratio.
        columns (list of `str`, optional): Defaults to None. List of columns.
        title (str, optional): Defaults to None. Plot title.
        plotting_context (str, optional): Defaults to 'notebook'. Seaborn plotting context.

    Returns:
        matplotlib.axes.Axes: axes.
    """

    with sns.plotting_context(plotting_context):
        odds_ratio = pd.Series(np.exp(coefs.ravel()), index=columns)
        if isinstance(top_n, int):
            odds_ratio = odds_ratio.loc[odds_ratio.sort_values(ascending=False)[
                :top_n].index]
        elif isinstance(top_n, float):
            odds_ratio = odds_ratio.loc[odds_ratio.sort_values(
                ascending=False).index]
            odds_ratio = odds_ratio[odds_ratio > top_n]
        ax = odds_ratio.plot.barh(color='C0')
        ax.set_xlabel('Change in odds ratio')
        ax.set_ylabel('Feature')
        if title:
            ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tick_params(axis='both', which='both', left=False, bottom=False)
        return ax
