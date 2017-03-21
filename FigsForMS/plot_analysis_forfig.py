import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_analysis_forfig(rows, columns, number, finbeats_subset,
                         finbeat_data, tracklist, zaxis='A', lines=True,
                         cutoff=False):
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
    count_n = 0  # start counting finbeats

    # find max initial speed for coloring by speed
    speeds = []
    for trial in finbeats_subset:
        speeds.append(tracklist[trial]['start_spd'])
    max_spd = max(speeds)

    # find x and y max and min axis limits
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
    cm = plt.get_cmap("plasma")

    # Set up the figure and choose an appropriate z-axis label
    ax1 = fig2.add_subplot(rows, columns, number, projection='3d')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Amplitude (cm)')
    if zaxis == 'V':
        ax1.set_zlabel('\nMax. Inst. Velocity(cm/s)')
    else:
        ax1.set_zlabel('\nMax. Acceleration (cm/s $^2$)')
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

            # find the maximum acceleration or velocity in that time range
            if zaxis == 'A':
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_a'][start:end].max()
                z_vals.append(zcolumn)
            elif zaxis == 'V':
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_v'][start:end].max()
                z_vals.append(zcolumn)
            else:  # If they fuck up, make it acceleration
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_a'][start:end].max()
                z_vals.append(zcolumn)

            # pull the initial speed and behavior
            init_spd = tracklist[trial]['start_spd']
            behavior_type = tracklist[trial]['behavior']
            if behavior_type == 'B':
                behavior = '*'
            elif behavior_type == 'A':
                behavior = '+'
            else:
                behavior = 'o'

            # add the point
            if cutoff == True and zaxis == 'A':
                z_max = 0.00005
            else:
                z_max = np.nanmax(z_vals)
            if zcolumn <= z_max and lines == True and zcolumn >= 0:
                p = ax1.plot(xs=[period, period], ys=[amplitude, amplitude],
                             zs=[0, zcolumn],
                             linestyle='solid', c=cm(init_spd / max_spd),
                             alpha=0.8, linewidth=0.5)
                p = ax1.scatter3D(xs=period,
                                  ys=amplitude,
                                  zs=zcolumn,
                                  zdir='z', s=50, marker=behavior,
                                  c=init_spd,
                                  cmap=cm, edgecolor='none', vmin=0,
                                  vmax=53)
                count_n += 1

    ax1.set_zlim3d(0, z_max)
    #cbar = plt.colorbar(p,shrink=0.7, pad = 0.1)
    #cbar.set_label('Initial Speed (cm/s)', rotation=270, labelpad=10)
    #if save == True:
        #plt.savefig(str(subset_name)+".svg", format="svg")
    #else:
        #plt.show()
    #print(count_n)

    return ax1