def get_p2o(p):
    """
    Probability to decimal odds
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    p: probability
    """
    return 1. / p


def get_o2p(o):
    """
    Decimal odds to probability
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    o: decimal odds
    """
    return 1. / o


def get_commission_rate(mm):
    """
    Market margin to commission rate
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    mm: market margin
    """
    return 1. - (1. / mm)


def get_commission_charged(o, r):
    """
    Commission charged = ((Stake * Odds Offered) - Stake) * Commission Rate
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    :param o: decimal odds
    :param r: commission rate
    :return:
    """
    return ((1. * o) - 1.) * r


def get_true_odds(o, r):
    """
    ((Stake * Odds Offered) - Commission Charged)  /  Stake
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    :param o: decimal odds
    :param r: commission rate
    :return:
    """
    return ((1. * o) - r) / 1.


def get_market_margin(p):
    """
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    Market Margin = sum(p)
    :param p: probabilities
    :return:
    """
    return sum(p)
