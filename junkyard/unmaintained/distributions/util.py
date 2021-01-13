"""
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120293/

How to use it:
import asyncio
dict_= dict()
asyncio.create_task(plot_scalers(dict_, df, iterable));

"""
import asyncio

import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, clear_output
from ipywidgets import Button, HBox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tabulate import tabulate


def wait_for_change(widget1, widget2, widget3):  # <------ Rename to widget1, and add widget2
    """
    https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#Button
    https://ipywidgets.readthedocs.io/en/latest/examples/Output%20Widget.html
    https://stackoverflow.com/questions/55244865/pause-jupyter-notebook-widgets-waiting-for-user-input
    """
    future = asyncio.Future()

    def getvalue(change):
        future.set_result(change.description)
        widget1.on_click(getvalue, remove=True)  # <------ Rename to widget1
        widget2.on_click(getvalue, remove=True)  # <------ New widget2
        widget3.on_click(getvalue, remove=True)  # <------ New widget3
        # we need to free up the binding to getvalue to avoid an IvalidState error
        # buttons don't support unobserve
        # so use `remove=True`

    widget1.on_click(getvalue)  # <------ Rename to widget1
    widget2.on_click(getvalue)  # <------ New widget2
    widget3.on_click(getvalue)  # <------ New widget3
    return future


def plot_rescaled(df, column_name, bins):
    """
    Plots rescaled distributions side by side for visual analysis
    :param df: DataFrame
    :param column_name: str, Column name to plot
    :param bins: int, number of histogram bins
    :return: str, user input value
    """

    nrows, ncols = 1, 3
    axs = plt.figure(figsize=(16, 4)).subplots(nrows, ncols)
    headers = ['Statistics', 'Original', 'StandardScaler', 'MinMaxScaler', 'RobustScaler']
    s = df[[column_name]].dropna()
    ss_ = pd.DataFrame(StandardScaler().fit_transform(s).flatten())
    mm_ = pd.DataFrame(MinMaxScaler().fit_transform(s).flatten())
    ro_ = pd.DataFrame(RobustScaler().fit_transform(s).flatten())
    table = [
        ['Mean', s.mean(), ss_.mean(), mm_.mean(), ro_.mean()],
        ['Std Dev', s.std(), ss_.std(), mm_.std(), ro_.std()],
        ['Variance', s.var(), ss_.var(), mm_.var(), ro_.var()],
        ['Min', s.min(), ss_.min(), mm_.min(), ro_.min()],
        ['Max', s.max(), ss_.max(), mm_.max(), ro_.max()],
        ['Skew', s.skew(), ss_.skew(), mm_.skew(), ro_.skew()],
        ['Kurtosis', s.kurtosis(), ss_.kurtosis(), mm_.kurtosis(), ro_.kurtosis()],
    ]
    print(tabulate(table, headers, tablefmt="psql"))
    ss_.plot(kind='hist', bins=bins, ax=axs[0], title='StandardScaler')
    mm_.plot(kind='hist', bins=bins, ax=axs[1], title='MinMaxScaler')
    ro_.plot(kind='hist', bins=bins, ax=axs[2], title='RobustScaler')
    plt.show()


async def plot_scalers(dict_, df, iterable, bins=100, out=widgets.Output(),
                       desc1='StandardScaler', desc2='MinMaxScaler', desc3='RobustScaler'):
    if dict_ is None:
        dict_ = dict()
    for c in iterable:
        with out:
            button1 = Button(description=desc1)  # <------ Rename to button1
            button2 = Button(description=desc2)  # <------ New button2 and description
            button3 = Button(description=desc3)  # <------ Rename to button1
            hb = HBox([button1, button2, button3])  # <----Display both buttons in an HBox
        display(hb)
        plot_rescaled(df, c, bins)
        x = await wait_for_change(button1, button2, button3)  # <---- Pass both buttons into the function
        dict_[c] = x
        clear_output(wait=True)
