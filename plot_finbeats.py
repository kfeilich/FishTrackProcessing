# Load all of the things
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rc('axes.formatter', useoffset=False)


def plot_finbeats(tracklist_subset, tracklist, finbeat_params):

    for trial in tracklist_subset:
        # shorten references for finbeat variables
        fb_peaktimes = finbeat_params[trial]['finbeat_peak_times']
        fb_effort = finbeat_params[trial]['finbeat_effort']
        fb_amplitude = finbeat_params[trial]['finbeat_amplitudes']
        fb_period = finbeat_params[trial]['finbeat_periods']

        fig = plt.figure()
        fig.suptitle(tracklist[trial]['sequence'])
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(fb_peaktimes, fb_effort, 'bo')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Tailbeat Effort (cm/s)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax2.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax2.tick_params('y', colors='r')

        ax3 = fig.add_subplot(2, 1, 2)
        ax3.plot(fb_peaktimes, fb_effort, 'bo')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tailbeat Effort (cm/s)', color='b')
        ax3.tick_params('y', colors='b')
        ax4 = ax3.twinx()
        ax4.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax4.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax4.tick_params('y', colors='r')
        plt.show()