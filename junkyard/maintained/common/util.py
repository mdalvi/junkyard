def get_matched_items(columns, match):
    """
    Gets the items in the columns which contain all the matches inside match
    :param columns: [str, str, str...]
    :param match: [str, str, str...]
    :return: list
    """
    return [c for c in columns if all(m in c for m in match)]
