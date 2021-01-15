import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tabulate import tabulate

from junkyard.maintained.common.util import get_parameter_vectors


def mixture_loss(nb_dimensions=2):
    """
    Computes the mean negative log-likelihood loss of y given the mixture parameters.
    https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca

    How to use:
    model.compile(loss=mixture_loss(nb_dimensions=2), ...)
    """

    def func(y_true, y_pred):
        alpha, mu, sigma = get_parameter_vectors(y_pred, nb_dimensions, 3)  # Unpack parameter vectors

        distr = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(probs=alpha),
            components_distribution=tfp.distributions.Normal(loc=mu, scale=sigma))

        log_likelihood = distr.log_prob(tf.transpose(y_true))  # Evaluate log-probability of y
        return -tf.reduce_mean(log_likelihood, axis=-1)

    return func


def heteroskedastic_loss(y_true, y_pred):
    """
    Heteroskedastic Loss (Quadratic Loss Function)
    https://nbviewer.jupyter.org/github/tensorchiefs/dl_book/blob/master/chapter_04/nb_ch04_04.ipynb
    https://nbviewer.jupyter.org/github/tensorchiefs/dl_book/blob/master/chapter_05/nb_ch05_01.ipynb
    :param y_true:
    :param y_pred:
    :return:
    """
    mu, sigma = get_parameter_vectors(y_pred, 1, 2)

    a = 1 / (tf.sqrt(2. * math.pi) * sigma)
    b1 = tf.square(mu - y_true)
    b2 = 2 * tf.square(sigma)
    b = b1 / b2
    return tf.reduce_mean(-tf.math.log(a) + b, axis=0)


def benters_loss(y_true, y_pred):
    """
    Sir William Benter's Loss
    The loss function can be used where each batch corresponds to 1 race
    https://doi.org/10.1142/9789812819192_0019
    https://stackoverflow.com/questions/42194051/filter-out-non-zero-values-in-a-tensor
    https://stackoverflow.com/questions/51405517/how-to-iterate-through-tensors-in-custom-loss-function
    """
    n = tf.shape(y_true)[0]
    mu, sigma = get_parameter_vectors(y_pred, 1, 2)

    y_true, y_speed = get_parameter_vectors(y_true, 1, 2)
    term1 = heteroskedastic_loss(y_true, y_pred)

    _, i = tf.math.top_k(tf.reshape(y_speed, [n, ]), k=n, sorted=True)
    mu = tf.gather(mu, indices=i)
    sigma = tf.gather(sigma, indices=i)

    tf_false = tf.cast(tf.zeros_like(y_true), dtype=tf.bool)
    tf_true = tf.cast(tf.ones_like(y_true), dtype=tf.bool)

    mu1 = mu[0]
    mu2 = tf.math.reduce_min(mu[1:])
    sigma1 = sigma[0]
    sigma2 = tf.boolean_mask(sigma, tf.where(mu == mu2, tf_true, tf_false))

    dist = tfp.distributions.Normal(0, 1)
    prob = dist.cdf(-(mu1 - mu2) / tf.math.sqrt(sigma1 + sigma2))
    term2 = -tf.math.log(prob + tf.keras.backend.epsilon())

    return term1 + term2


def binary_cross_entropy_payoffs_and_regret(y_true, y_pred):
    """
    BinaryCrossEntropy with PayOffs and Regret
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon()) * (
            tf.reduce_max(payoffs) + tf.reduce_min(payoffs) - tf.math.abs(payoffs))  # Cancels out when target is 1
    term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) * tf.math.abs(
        payoffs)  # Cancels out when target is 0

    return -tf.reduce_mean(term_0 + term_1, axis=1)


def binary_cross_entropy_payoffs_quadratic(y_true, y_pred):
    """
    BinarySquaredCrossEntropy with Payoffs (Quadratic Loss Function)
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = (1 - y_true) * y_pred  # Cancels out when target is 1
    term_1 = y_true * (1 - y_pred)  # Cancels out when target is 0

    return tf.math.square(tf.math.abs((term_0 + term_1) * tf.math.abs(payoffs)))


def binary_cross_entropy_payoffs_high_penalty(y_true, y_pred):
    """
    BinaryCrossEntropy with Payoffs (Exponential Loss Function)
    tf.reduce_max(payoffs * y_true) - ... Ensures higher penalty for the batch with higher payoffs
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())  # Cancels out when target is 1
    term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())  # Cancels out when target is 0

    return -tf.reduce_mean((term_0 + term_1) * tf.math.abs(payoffs) * tf.reduce_max(payoffs * y_true), axis=1)


def binary_cross_entropy_payoffs_rbartel(y_true, y_pred):
    """
    Richard Bartel's Custom Loss with ReLu (Linear Loss Function)
    https://medium.com/vantageai/beating-the-bookies-with-machine-learning-7b429a0b5980
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = (1 - y_true) * tf.math.abs(payoffs) * tf.nn.relu(
        tf.math.abs(payoffs) * y_pred)  # Cancels out when target is 1
    term_1 = y_true * tf.math.abs(payoffs) * tf.nn.relu(
        tf.math.abs(payoffs) * (1 - y_pred))  # Cancels out when target is 0

    return tf.reduce_mean(term_0 + term_1, axis=1)


def binary_cross_entropy_payoffs_alexr_high_penalty(y_true, y_pred):
    """
    Alex R.'s BinaryCrossEntropy with Payoffs (Quadratic Loss Function)
    tf.reduce_max(payoffs * y_true) - ... Ensures higher penalty for the batch with higher payoffs
    https://stats.stackexchange.com/questions/318234/the-best-way-to-maximize-a-payoff-based-on-a-binary-decision-with-neural-network
    Note: This loss function does not converge at zero.
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = tf.reduce_sum((1 - y_true) * tf.math.abs(payoffs) * (1 - y_pred), axis=1)  # Cancels out when target is 1
    term_1 = tf.reduce_sum(y_true * tf.math.abs(payoffs) * y_pred, axis=1)  # Cancels out when target is 0

    return tf.math.square(tf.math.abs(tf.reduce_max(payoffs * y_true) - term_1 - term_0))


def binary_cross_entropy_payoffs_alexr(y_true, y_pred):
    """
    Alex R.'s BinaryCrossEntropy with Payoffs (Linear Loss Function)
    https://stats.stackexchange.com/questions/318234/the-best-way-to-maximize-a-payoff-based-on-a-binary-decision-with-neural-network
    Note: This loss function does not converge at zero.
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = tf.reduce_sum((1 - y_true) * tf.math.abs(payoffs) * (1 - y_pred), axis=1)  # Cancels out when target is 1
    term_1 = tf.reduce_sum(y_true * tf.math.abs(payoffs) * y_pred, axis=1)  # Cancels out when target is 0

    return tf.math.square(tf.math.abs(tf.reduce_max(payoffs) - term_1 - term_0))


def binary_cross_entropy_payoffs(y_true, y_pred):
    """
    BinaryCrossEntropy with Payoffs (Exponential Loss Function)
    Higher the payoffs, higher the gradient (symmetric)
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())  # Cancels out when target is 1
    term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())  # Cancels out when target is 0

    return -tf.reduce_mean((term_0 + term_1) * tf.math.abs(payoffs), axis=1)


def binary_cross_entropy(y_true, y_pred):
    """
    BinaryCrossEntropy Loss (Exponential Loss Function)
    """

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    term_0 = (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())  # Cancels out when target is 1
    term_1 = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())  # Cancels out when target is 0

    return -tf.reduce_mean(term_0 + term_1, axis=1)


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
