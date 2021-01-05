import warnings

import numpy as np
from scipy.stats import norm


def overlap_solver(mean1, std_dev1, mean2, std_dev2):
    """
    # The probability overlap solver
    # Determines the area (in percentage) overlap between two normal distributions
    https://stackoverflow.com/questions/32551610/overlapping-probability-of-two-normal-distribution-with-scipy
    :param mean1:
    :param std_dev1:
    :param mean2:
    :param std_dev2:
    :return:
    """
    a = 1. / (2. * std_dev1 ** 2) - 1. / (2. * std_dev2 ** 2)
    b = mean2 / (std_dev2 ** 2) - mean1 / (std_dev1 ** 2)
    c = mean1 ** 2 / (2 * std_dev1 ** 2) - mean2 ** 2 / (2 * std_dev2 ** 2) - np.log(std_dev2 / std_dev1)
    result = np.roots([a, b, c])

    size_ = 1000

    x = np.random.normal(mean1, std_dev1, size=(size_,))
    y = np.random.normal(mean2, std_dev2, size=(size_,))

    x = np.linspace(min(min(x), min(y)), max(max(x), max(y)), size_)

    overlap, lower, upper = 0., min(x), max(x)

    if len(result) == 0:  # completely overlapping
        overlap = 1.

    elif len(result) == 1:  # one point of contact

        r = result[0]
        if mean1 > mean2:
            tm, ts = mean2, std_dev2
            mean2, std_dev2 = mean1, std_dev1
            mean1, std_dev1 = tm, ts
        if r < lower:  # point of contact is less than the lower boundary. order: r-l-u
            overlap = (norm.cdf(upper, mean1, std_dev1) - norm.cdf(lower, mean1, std_dev1))
        elif r < upper:  # point of contact is more than the upper boundary. order: l-u-r
            overlap = (norm.cdf(r, mean2, std_dev2) - norm.cdf(lower, mean2, std_dev2)) + (
                    norm.cdf(upper, mean1, std_dev1) - norm.cdf(r, mean1, std_dev1))
        else:  # point of contact is within the upper and lower boundaries. order: l-r-u
            overlap = (norm.cdf(upper, mean2, std_dev2) - norm.cdf(lower, mean2, std_dev2))

    elif len(result) == 2:  # two points of contact
        r1 = result[0]
        r2 = result[1]
        if r1 > r2:
            temp = r2
            r2 = r1
            r1 = temp
        if std_dev1 > std_dev2:
            tm, ts = mean2, std_dev2
            mean2, std_dev2 = mean1, std_dev1
            mean1, std_dev1 = tm, ts
        if r1 < lower:
            if r2 < lower:  # order: r1-r2-l-u
                overlap = (norm.cdf(upper, mean1, std_dev1) - norm.cdf(lower, mean1, std_dev1))
            elif r2 < upper:  # order: r1-l-r2-u
                overlap = (norm.cdf(r2, mean2, std_dev2) - norm.cdf(lower, mean2, std_dev2)) + (
                        norm.cdf(upper, mean1, std_dev1) - norm.cdf(r2, mean1, std_dev1))
            else:  # order: r1-l-u-r2
                overlap = (norm.cdf(upper, mean2, std_dev2) - norm.cdf(lower, mean2, std_dev2))
        elif r1 < upper:
            if r2 < upper:  # order: l-r1-r2-u
                overlap = (norm.cdf(r1, mean1, std_dev1) - norm.cdf(lower, mean1, std_dev1)) + (
                        norm.cdf(r2, mean2, std_dev2) - norm.cdf(r1, mean2, std_dev2)) + (
                                  norm.cdf(upper, mean1, std_dev1) - norm.cdf(r2, mean1, std_dev1))
            else:  # order: l-r1-u-r2
                overlap = (norm.cdf(r1, mean1, std_dev1) - norm.cdf(lower, mean1, std_dev1)) + (
                        norm.cdf(upper, mean2, std_dev2) - norm.cdf(r1, mean2, std_dev2))
        else:  # l-u-r1-r2
            overlap = (norm.cdf(upper, mean1, std_dev1) - norm.cdf(lower, mean1, std_dev1))

    if isinstance(overlap, complex):
        warnings.warn(f"Complex number with parameters {mean1}, {std_dev1}, {mean2}, {std_dev2}", UserWarning)
        return np.nan

    return overlap
