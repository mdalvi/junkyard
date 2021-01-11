import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


def plot_public_accuracy_vs_time(df_, time_id_column, is_public_win_column, is_win_column,
                                 title='PUBLIC ACCURACY VS TIME'):
    """
    https://stackoverflow.com/questions/39714724/pandas-plot-x-axis-tick-frequency-how-can-i-show-more-ticks
    """
    assert_str = 'Null values observed in time_id_column or is_win_column'
    assert (df_[time_id_column].isnull().sum() == 0 and df_[is_win_column].isnull().sum() == 0), assert_str

    df_ = df_[[time_id_column, is_public_win_column, is_win_column]].dropna(subset=[is_public_win_column]).copy()
    df_['year'] = pd.DatetimeIndex(df_[time_id_column]).year
    df_['month'] = pd.DatetimeIndex(df_[time_id_column]).month

    def func(df_):
        exp = df_[(df_[is_public_win_column] == 1)].shape[0]
        act = df_[(df_[is_public_win_column] == 1) & (df_[is_win_column] == 1)].shape[0]

        df_['pub_acc'] = act / exp
        return df_

    df_grp = df_[['year', 'month', is_public_win_column, is_win_column]].dropna().groupby(
        ['year', 'month']).apply(func).reset_index(drop=True)
    df_grp = df_grp.drop([is_public_win_column, is_win_column], axis=1).drop_duplicates().reset_index(drop=True)
    df_grp['timescale'] = pd.to_datetime(df_grp[['year', 'month']].apply(lambda row: f"{row[0]}-{row[1]}", axis=1))

    print(tabulate(df_grp[['pub_acc']].describe(), headers='keys', tablefmt='psql'))
    ax = df_grp.plot(x='timescale', y='pub_acc', figsize=(16, 4), x_compat=True, title=title)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.show()


def get_unbiased_prob_estimates(df, race_id_column, prob_column, is_win_column, title='UNBIASED PROBABILITY ESTIMATES'):
    """
    https://www.worldscientific.com/doi/abs/10.1142/9789812819192_0019
    https://www.semanticscholar.org/paper/Computer-Based-Horse-Race-Handicapping-and-Wagering-Benter/2ea3ed4fa5ea9645614d76dd0a79201740949566
    :param df:
    :param race_id_column:
    :param prob_column:
    :param is_win_column:
    :param title:
    :return:
    """
    ranges = [
        (0.000, 0.010),
        (0.010, 0.025),
        (0.025, 0.050),
        (0.050, 0.100),
        (0.100, 0.150),
        (0.150, 0.200),
        (0.200, 0.250),
        (0.250, 0.300),
        (0.300, 0.400),
    ]

    headers = ['range', 'n', 'exp.', 'act.', 'act. - exp.']
    table = list()
    df_ = df[[race_id_column, prob_column, is_win_column]].dropna(subset=[prob_column]).copy()

    assert_str = 'Null values observed in race_id_column or is_win_column'
    assert (df[race_id_column].isnull().sum() == 0 and df[is_win_column].isnull().sum() == 0), assert_str
    for a, b in ranges:
        n = df_[(df_[prob_column] > a) & (df_[prob_column] <= b)].shape[0]
        if n == 0.:
            act, exp = 0., 0.
        else:
            act = df_[(df_[prob_column] > a) & (df_[prob_column] <= b) & (df_[is_win_column] == 1)].shape[0] / n
            exp = df_[(df_[prob_column] > a) & (df_[prob_column] <= b)][prob_column].mean()
        table.append(
            [f"{a:.3f}-{b:.3f}", n, f"{round(exp, 3):.3f}", f"{round(act, 3):.3f}", f"{round(act - exp, 3):.3f}"])

    n = df_[(df_[prob_column] > ranges[-1][1])].shape[0]
    if n == 0:
        act, exp = 0., 0.
    else:
        act = df_[(df_[prob_column] > ranges[-1][1]) & (df_[is_win_column] == 1)].shape[0] / n
        exp = df_[(df_[prob_column] > ranges[-1][1])][prob_column].mean()
    table.append(
        [f">{ranges[-1][1]:.3f}", n, f"{round(exp, 3):.3f}", f"{round(act, 3):.3f}", f"{round(act - exp, 3):.3f}"])

    print(title)
    print(tabulate(table, headers, tablefmt="psql"))
    print(f'# races = {df_[race_id_column].unique().size} , # horses = {df_.shape[0]}')
    print('')
    print('range = the range of estimated probabilities')
    print('n = the number of horses falling within a range')
    print('exp. = the mean expected probability')
    print('act. = the actual win frequency observed')
    print('act. - exp. = the discrepancy (+ or -)')
