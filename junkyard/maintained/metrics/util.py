import tensorflow as tf

from junkyard.maintained.common.util import get_parameter_vectors


def profit(y_true, y_pred):
    """
    Metric - Profit
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    return tf.reduce_sum(y_true * tf.math.round(y_pred) * payoffs + tf.math.round(y_pred) * -1.)


def confusion(y_true, y_pred):
    """
    Confusion Matrix
    https://github.com/dickreuter/betfair-horse-racing/blob/master/horse_racing/neural_networks/custom_optimization.py
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true, payoffs = get_parameter_vectors(y_true, 1, 2)

    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

    y_pred_pos = tf.math.round(tf.clip_by_value(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = tf.math.round(tf.clip_by_value(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp_ = tf.reduce_sum(y_pos * y_pred_pos) / (tf.keras.backend.epsilon() + tf.reduce_sum(y_pos))
    tn_ = tf.reduce_sum(y_neg * y_pred_neg) / (tf.keras.backend.epsilon() + tf.reduce_sum(y_neg))
    fn_ = tf.reduce_sum(y_pos * y_pred_neg) / (tf.keras.backend.epsilon() + tf.reduce_sum(y_neg))
    fp_ = tf.reduce_sum(y_neg * y_pred_pos) / (tf.keras.backend.epsilon() + tf.reduce_sum(y_pos))

    return tp_, tn_, fn_, fp_


def tp(y_true, y_pred):
    """
    True positives (percentage)
    """
    tp_, _, _, _ = confusion(y_true, y_pred)
    return tp_


def tn(y_true, y_pred):
    """
    True negatives (percentage)
    """
    _, tn_, _, _ = confusion(y_true, y_pred)
    return tn_


def fn(y_true, y_pred):
    """
    False negatives (percentage)
    """
    _, _, fn_, _ = confusion(y_true, y_pred)
    return fn_


def fp(y_true, y_pred):
    """
    False positives (percentage)
    """
    _, _, _, fp_ = confusion(y_true, y_pred)
    return fp_
