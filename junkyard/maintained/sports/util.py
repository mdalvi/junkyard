def get_p2o(p):
    """
    Probability to decimal odds
    https://en.wikipedia.org/wiki/Parimutuel_betting
    http://www.aussportsbetting.com/guide/sports-betting-arbitrage/
    http://www.aussportsbetting.com/tools/online-calculators/arbitrage-calculator/
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    p: probability
    :return: float
    """
    return 1. / p


def get_o2p(o):
    """
    Decimal odds to probability
    https://en.wikipedia.org/wiki/Parimutuel_betting
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    o: decimal odds
    :return: float
    """
    return 1. / o


def get_commission_rate(mm):
    """
    Market margin to commission rate
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    mm: market margin
    :return: float
    """
    return 1. - (1. / mm)


def get_commission_charged(o, r):
    """
    Commission charged = ((Stake * Odds Offered) - Stake) * Commission Rate
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    :param o: decimal odds
    :param r: commission rate
    :return: float
    """
    return ((1. * o) - 1.) * r


def get_true_odds(o, r):
    """
    ((Stake * Odds Offered) - Commission Charged)  /  Stake
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    :param o: decimal odds
    :param r: commission rate
    :return: float
    """
    return ((1. * o) - r) / 1.


def get_market_margin(p):
    """
    Market Margin = sum(p)
    https://www.bettingexpert.com/academy/advanced-betting-theory/calculating-bookmaker-commission
    :param p: probabilities
    :return: float
    """
    return sum(p)


def get_payoffs(o, bet_unit=1) -> float:
    """
    Returns payoffs from decimal odds
    :param o: Decimal odds
    :param bet_unit: Betting unit
    :return: float
    """
    return round(bet_unit * o, 0) - bet_unit
