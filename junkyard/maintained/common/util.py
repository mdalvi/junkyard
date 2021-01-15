def get_matched_items(columns, match):
    """
    Gets the items in the columns which contain all the matches inside match
    :param columns: [str, str, str...]
    :param match: [str, str, str...]
    :return: list
    """
    return [c for c in columns if all(m in c for m in match)]


def get_parameter_vectors(parameter_vector, nb_dimensions, nb_parameters):
    """
    Returns an unpacked list of parameters vectors.
    https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
    """
    return [parameter_vector[:, i * nb_dimensions:(i + 1) * nb_dimensions] for i in range(nb_parameters)]
