import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


def plot_keras_history(history, metric=None, metric_title="Metric"):
    """
    Plots keras training history (with optional metric)
    :param history: keras.history
    :param metric: str, metric name
    :param metric_title: str, metric name (for tabulate)
    :return: None
    """
    df_ = pd.DataFrame(history.history)

    if metric is None:
        headers = ['Statistics', 'Loss', 'Validation Loss']
        extend_min, extend_max, extend_final = [], [], []
        axs = plt.figure(figsize=(6, 4)).subplots(1, 1)
    else:
        headers = ['Statistics', 'Loss', 'Validation Loss', metric_title, f'Validation {metric_title}']
        extend_min = [df_[metric].min(), df_[f'val_{metric}'].min()]
        extend_max = [df_[metric].max(), df_[f'val_{metric}'].max()]
        extend_final = [df_[metric].iloc[-1], df_[f'val_{metric}'].iloc[-1]]
        axs = plt.figure(figsize=(14, 4)).subplots(1, 2)

    table = [
        ['Min', df_['loss'].min(), df_['val_loss'].min()] + extend_min,
        ['Max', df_['loss'].max(), df_['val_loss'].max()] + extend_max,
        ['Final', df_['loss'].iloc[-1], df_['val_loss'].iloc[-1]] + extend_final,
    ]
    print(tabulate(table, headers, tablefmt="psql"))

    if metric is None:
        df_[['loss', 'val_loss']].plot(ax=axs)
    else:
        df_[['loss', 'val_loss']].plot(ax=axs[0])
        df_[[metric, f'val_{metric}']].plot(ax=axs[1])

    plt.show()
