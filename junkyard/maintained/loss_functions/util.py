import tensorflow as tf


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
