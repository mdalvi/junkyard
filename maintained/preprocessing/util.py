def rescale(value, actual_range=(0, 1), normal_range=(0, 1)):
    """
    https://github.com/mdalvi/evolving-networks/blob/master/evolving_networks/math_util.py#L24
    """
    epsilon_ = 1e-8
    act_min, act_max = actual_range
    nor_min, nor_max = normal_range

    return ((value - act_min) / (act_max - act_min + epsilon_)) * (nor_max - nor_min) + nor_min
