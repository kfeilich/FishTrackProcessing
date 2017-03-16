import numpy as np
import pandas as pd

def make_subset(group_by, identifier, tracklist):
    """
    Args:


    Returns:

    """
    subset = []
    for i in tracklist.keys():
        if tracklist[i][group_by] == identifier:
            subset.append(i)

    return subset