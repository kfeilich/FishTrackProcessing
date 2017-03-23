import peakutils
import numpy as np
import matplotlib.pyplot as plt


def plot_trace_forfig(rows, columns, number, trial, tracklist):
    """Plots finbeats in (period, amplitude, acceleration) space.

        This function takes finbeat data from a specified output of
        finbeat_calc(), and plots each individual finbeat in (period,
        amplitude, maximum acceleration) space. The finbeat_data argument
        specifies whether the finbeats to be plotted come from peak-to-peak
        or trough-to-trough calculations. The maximum acceleration is the
        maximum acceleration between the finbeat start and finbeat end
        times. The number of total finbeats is printed at the end.

        Args:
            subset_name (string): some string identifying what's in your
                                    subset, to be used as the plot title
            finbeats_subset (list): a list of strings with the trial
                                names of the desired trials from finbeats.
                                 Note: The list (even of a single
                                 element) must be contained in square
                                 brackets. You'll probably want to use
                                 the subset generating function:
                                 make_subset()
            finbeat_data (dict): use either finbeat_byP to do analysis
                                on finbeats as defined by peaks first,
                                or finbeat_byT to use finbeats defined by
                                troughs first. These must be created
                                beforehand by the function finbeat_calc()
            zaxis (str): must be of value "A" or "V". Indicates whether to plot
                            acceleration or velocity.
            tracklist (dict): the compiled position, velocity,
                              and acceleration data for all trials
                               produced by extract_data()
            lines (Bool): if True, adds lines up from x-y plane to z_value
            cutoff (Bool): if True, cuts off z axis at hard-coded maximum value
            save (Bool): if True, saves to svg instead of printing to screen

        Returns:
            Nothing
        """

    raw_data = tracklist[trial]['data']['pt2y']
    behavior = tracklist[trial]['behavior']
    init_speed = tracklist[trial]['start_spd_BLs']
    base = peakutils.baseline(raw_data, 3)  # Find bkgrd trend
    raw_data = raw_data-base
    time = raw_data.index.values