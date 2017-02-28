# Load all of the things
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rc('axes.formatter', useoffset=False)


def plot_accel(tracklist_subset, tracklist):
    """Plot _____ from data as produced
            by process_points_pandas.py.

           Parameters
           ----------
           tracklist_subset : 1D array_like
               List of strings indicating sequence names of desired trials.
           tracklist : pandas dataframe
               tracklist dataframe produced by process_points_pandas.py

           Returns
           -------

           Notes
           -----


           References
           ----------

           Examples
           --------
           """
    tracklist_subset = input("Input Desired Trial Subset as List")
    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)
        data = tracklist[trial]['data']
        # filename = str(trial) + '.pdf'

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist[trial]['behavior'])
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(data.index.values, data['pt2y_smth'], 'bo')
        ax1.set_ylabel('Tail Excursion (cm)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(data.index.values, data['pt1x_a'], 'r.')
        ax2.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax1.set_xlabel('Time (s)')
        ax2.tick_params('y', colors='r')
        plt.axhline(0, color='r', linewidth=4, linestyle='dashed')
        # plt.savefig(filename)
        plt.show()