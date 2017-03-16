import numpy as np
import pandas as pd

def make_subset(group_by, identifier, tracklist):
    """Generates lists suitable for use as subsets in other fns

    This function produces lists of trials from tracklist grouped by
    factors of interest, that are suitable for use in other
    functions, including finbeat_calc(), plot_track(), plot_accel(),
    check_plots(), and plot_analysis() in particular. Facilitates
    easy subset creation by species or fish.

    Args:
        group_by (str): the factor to search for your identifier of
                        interest, usually 'fish' or 'species'
        identifier (str): the specific identifier you want to index,
                           e.g. 'Bass' or 'BTrout' if group_by =
                          'species'; or 'Bass1' or 'BTrout2' if
                          group_by = 'fish'.
        tracklist (dict): the compiled position, velocity,
                          and acceleration data for all trials
                           produced by extract_data()

    Returns:
        subset (list): a list of strings containing the trial names
                        of trials matching the specified identifier,
                        which can be used in other functions.
    """
    subset = []
    for i in tracklist.keys():
        if tracklist[i][group_by] == identifier:
            subset.append(i)

    return subset