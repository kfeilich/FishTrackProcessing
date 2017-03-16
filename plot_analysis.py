import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_analysis(subset_name, finbeats_subset, finbeat_data,
                  tracklist):
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
        tracklist (dict): the compiled position, velocity,
                          and acceleration data for all trials
                           produced by extract_data()

    Returns:
        Nothing
    """
    count_n = 0  # start counting finbeats

    #find max initial speed for coloring by speed
    speeds = []
    for trial in finbeats_subset:
        speeds.append(tracklist[trial]['start_spd'])
    max_spd = max(speeds)

    # find max and min axis limits
    x_vals = []
    y_vals = []
    z_vals = []
    for trial in finbeats_subset:
        for finbeat in finbeat_data[trial].index.values:
            x_vals.append(finbeat_data[trial]['period'][finbeat])
            y_vals.append(finbeat_data[trial]['amplitude'][finbeat])

    x_max = np.nanmax(x_vals)
    y_max = np.nanmax(y_vals)


    # Pull a colormap
    cm = plt.get_cmap("YlOrRd")

    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(13)
    fig.suptitle(subset_name)
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Amplitude (cm)')
    ax1.set_zlabel('Maximum Accel (cm/s2)')
    ax1.set_xlim3d(0, x_max)
    ax1.set_ylim3d(0, y_max)


    # for each trial of interest
    for trial in finbeats_subset:
        # for each finbeat within that trial
        for finbeat in finbeat_data[trial].index.values:
            # get the period
            # period_mask = finbeat_data[trial]['period'].loc[finbeat]
            period = finbeat_data[trial]['period'][finbeat]

            # get the amplitude
            # amplitude_mask = finbeat_data[trial]['amplitude'].loc[
            # finbeat]
            amplitude = finbeat_data[trial]['amplitude'][finbeat]

            # get the start time
            # start_mask = finbeat_data[trial]['time'].loc[finbeat]
            start = finbeat_data[trial]['time'][finbeat]
            # get the end time
            # end_mask = finbeat_data[trial]['endtime'].loc[finbeat]
            end = finbeat_data[trial]['endtime'][finbeat]

            # find the maximum acceleration in that time range
            accel = tracklist[trial]['data'][
                        'pt1_net_a'][start:end].max()
            z_vals.append(accel)

            # pull the initial speed
            init_spd = tracklist[trial]['start_spd']

            # add the point
            p=ax1.scatter3D(xs=period,
                          ys=amplitude,
                          zs=accel,
                          zdir='z', s=20, marker='o', c=init_spd,
                          cmap = cm, edgecolor='none', vmin=0,
                          vmax=max_spd)
            count_n += 1
    z_max = np.nanmax(z_vals)
    ax1.set_zlim3d(0, z_max)
    cbar = plt.colorbar(p)
    cbar.set_label('Initial Speed (cm/s)', rotation =270)
    plt.show()
    print(count_n)