# Load all of the things
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rc('axes.formatter', useoffset=False)


def plot_net_accel(tracklist_subset, tracklist):
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
    #tracklist_subset = input("Input Desired Trial Subset as List")
    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)
        data = tracklist[trial]['data']
        # filename = str(trial) + '.pdf'

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist[trial]['behavior'])
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(tracklist[trial]['data'].index.values,
                 tracklist[trial]['data']['pt1_net_v'], 'bo')
        ax1.set_ylabel('Pt1 Net Velocity(cm/s)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(tracklist[trial]['data'].index.values,
                 tracklist[trial]['data']['pt1_net_a'], 'r.')
        ax2.set_ylabel('Pt1 Net Accel (cm/s2)', color='r')
        ax1.set_xlabel('Time (s)')
        ax2.tick_params('y', colors='r')
        plt.axhline(0, color='r', linewidth=4, linestyle='dashed')
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.plot(tracklist[trial]['data'].index.values,
                 tracklist[trial]['data']['pt2y_smth], 'bo')
        ax3.set_ylabel('Pt2 Net Velocity(cm/s)', color='b')
        ax3.tick_params('y', colors='b')
        ax4 = ax3.twinx()
        ax4.plot(tracklist[trial]['data'].index.values,
                 tracklist[trial]['data']['pt2_net_a'], 'r.')
        ax4.set_ylabel('Pt2 Net Accel (cm/s2)', color='r')
        ax3.set_xlabel('Time (s)')
        ax4.tick_params('y', colors='r')
        plt.axhline(0, color='r', linewidth=4, linestyle='dashed')
        # plt.savefig(filename)
        plt.show()