import tensorflow as tf


def nnelu(ip):
    """
    Computes the Non-Negative Exponential Linear Unit
    https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(ip))


tf.keras.utils.get_custom_objects().update({'nnelu': nnelu})
