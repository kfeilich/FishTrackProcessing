import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_analysis(subset_name, finbeats_subset, finbeat_info,
                  tracklist):
    """
    Args:
        subset_name (string): some string identifying what's in your
                                subset, to be used as plot title
        finbeats_subset (list): a list of strings with the trial
                            names of the desired trials from finbeats.
                             Note: The list (even of a single
                             element) must be contained in square
                             brackets.
                             Also note: You'll probably want to use
                             one of the subset generating functions,
                             make_subset()
        finbeat_info (dict): use either finbeat_byP to do analysis
                            on finbeats as defined by peaks first,
                            or finbeat_byT to use finbeats defined by
                            troughs first.
        tracklist (dict)

    Returns:
        finbeats[trial_name] (dict):
    """
    count_n = 0  # start counting finbeats
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.suptitle(subset_name)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Amplitude (cm)')
    ax1.set_zlabel('Maximum Accel (cm/s2)')

    # for each trial of interest
    for trial in finbeats_subset:
        # for each finbeat within that trial
        for finbeat in finbeat_info[trial].index.values:
            # get the period
            # period_mask = finbeat_info[trial]['period'].loc[finbeat]
            period = finbeat_info[trial]['period'][finbeat]

            # get the amplitude
            # amplitude_mask = finbeat_info[trial]['amplitude'].loc[
            # finbeat]
            amplitude = finbeat_info[trial]['amplitude'][finbeat]

            # get the start time
            # start_mask = finbeat_info[trial]['time'].loc[finbeat]
            start = finbeat_info[trial]['time'][finbeat]
            # get the end time
            # end_mask = finbeat_info[trial]['endtime'].loc[finbeat]
            end = finbeat_info[trial]['endtime'][finbeat]

            # find the maximum acceleration in that time range
            accel = tracklist[trial]['data'][
                        'pt1_net_a'][start:end].max()

            # add the point
            ax1.scatter3D(xs=period,
                          ys=amplitude,
                          zs=accel,
                          zdir='z', s=20, marker='o', c='black',
                          edgecolor='none')
            count_n += 1

    plt.show()
    print(count_n)