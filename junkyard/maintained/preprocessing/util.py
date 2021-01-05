def get_rescale(value, actual_range=(0, 1), normal_range=(0, 1)):
    """
    https://github.com/mdalvi/evolving-networks/blob/master/evolving_networks/math_util.py#L24
    :param value: value from the actual scale to normalize
    :param actual_range: actual range of the scale
    :param normal_range: normalized range of the scale
    :return: float
    """
    epsilon_ = 1e-8
    act_min, act_max = actual_range
    nor_min, nor_max = normal_range

    return ((value - act_min) / (act_max - act_min + epsilon_)) * (nor_max - nor_min) + nor_min


def get_embedding_size(n_cat: int) -> int:
    """
    Determines the embedding vector size for number of categories
    https://github.com/fastai/fastai/blob/96c5927648ecf83f0bc9ab601f672d3c0ffe0059/fastai/tabular/data.py#L13
    :param n_cat: number of categories
    :return: int
    """
    return min(600, round(1.6 * n_cat ** 0.56))


def get_categorical_inputs_for_dl(x_arr):
    """
    https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
    :param x_arr: numpy array of categorical data of shape (n_samples, n_columns)
    :return: list
    """
    return [x_arr[:, i] for i in range(x_arr.shape[1])]