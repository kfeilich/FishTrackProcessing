import numpy as np
import pandas as pd

def make_subset(group_by1, identifier1, tracklist, group_by2=None,
                identifier2=None, group_by3=None, identifier3=None):
    """Generates lists suitable for use as subsets in other fns

    This function produces lists of trials from tracklist grouped by
    factors of interest, that are suitable for use in other
    functions, including finbeat_calc(), plot_track(), plot_accel(),
    check_plots(), and plot_analysis() in particular. Facilitates
    easy subset creation by species or fish.

    Args:
        group_by1 (str): the factor to search for your identifier of
                        interest, usually 'fish' or 'species'
        identifier1 (str): the specific identifier you want to index,
                           e.g. 'Bass' or 'BTrout' if group_by =
                          'species'; or 'Bass1' or 'BTrout2' if
                          group_by = 'fish'.
        tracklist (dict): the compiled position, velocity,
                          and acceleration data for all trials
                           produced by extract_data()
        group_by2
        identifier2
        group_by3
        identifier3

    Returns:
        subset (list): a list of strings containing the trial names
                        of trials matching the specified identifier,
                        which can be used in other functions.
    """
    subset = []

    for i in tracklist.keys():
        if tracklist[i][group_by1] == identifier1:
            subset.append(i)

    if group_by2 != None:
        subset2 = list(subset)
        for j in subset:
            if tracklist[j][group_by2] != identifier2:
                subset2.remove(j)
    else:
        return subset

    if group_by3 != None:
        subset3 = list(subset2)
        for k in subset2:
            if tracklist[k][group_by3] != identifier3:
                subset3.remove(k)
    else:
        return subset2

    return subset3