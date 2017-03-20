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

        # Subplot 1: Effort and X-Acceleration on Time
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(fb_peaktimes, fb_effort, 'bo')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Tailbeat Effort (cm/s)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax2.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax2.tick_params('y', colors='r')

        # Sublplot 2: Effort and X-Velocity on Time
        ax3 = fig.add_subplot(3, 2, 2)
        ax3.plot(fb_peaktimes, fb_effort, 'bo')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tailbeat Effort (cm/s)', color='b')
        ax3.tick_params('y', colors='b')
        ax4 = ax3.twinx()
        ax4.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax4.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax4.tick_params('y', colors='r')

        # Subplot 3: Amplitude and X-Acceleration on Time
        ax5 = fig.add_subplot(3, 2, 3)
        ax5.plot(fb_peaktimes, fb_amplitude, 'bo')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Tailbeat Amplitude (cm)', color='b')
        ax5.tick_params('y', colors='b')
        ax6 = ax5.twinx()
        ax6.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax6.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax6.tick_params('y', colors='r')

        # Subplot 4: Amplitude and X-Velocity on Time
        ax7 = fig.add_subplot(3, 2, 4)
        ax7.plot(fb_peaktimes, fb_amplitude, 'bo')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Tailbeat Amplitude (cm)', color='b')
        ax7.tick_params('y', colors='b')
        ax8 = ax7.twinx()
        ax8.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax8.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax8.tick_params('y', colors='r')

        # Subplot 5: Period and X-Acceleration on Time
        ax9 = fig.add_subplot(3, 2, 5)
        ax9.plot(fb_peaktimes, fb_period, 'bo')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Tailbeat Period (s)', color='b')
        ax9.tick_params('y', colors='b')
        ax10 = ax9.twinx()
        ax10.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax10.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax10.tick_params('y', colors='r')

        # Subplot 6: Period and X-Velocity on Time
        ax11 = fig.add_subplot(3, 2, 6)
        ax11.plot(fb_peaktimes, fb_period, 'bo')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Tailbeat Period (s)', color='b')
        ax11.tick_params('y', colors='b')
        ax12 = ax11.twinx()
        ax12.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax12.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax12.tick_params('y', colors='r')
        plt.show()