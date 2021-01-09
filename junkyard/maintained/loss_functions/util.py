import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tabulate import tabulate


def get_two_splits(y_true):
    """
    Divides y_true into two vectors
    :param y_true:
    :return:
    """
    child_y = y_true[:, 1]
    child_y = tf.expand_dims(child_y, 1)
    y_true = y_true[:, 0]
    y_true = tf.expand_dims(y_true, 1)
    return y_true, child_y


def get_three_splits(y_true):
    """
    Divides y_true into three vectors
    :param y_true:
    :return:
    """
    child_y_2 = y_true[:, 2]
    child_y_2 = tf.expand_dims(child_y_2, 1)
    child_y_1 = y_true[:, 1]
    child_y_1 = tf.expand_dims(child_y_1, 1)
    y_true = y_true[:, 0]
    y_true = tf.expand_dims(y_true, 1)
    return y_true, child_y_1, child_y_2


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
