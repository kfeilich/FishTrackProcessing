# Load all of the things
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rc('axes.formatter', useoffset=False)


def check_plots(tracklist_subset, tracklist):
    """Plot some diagnostics from trial data produced by extract_data()
    
    This is just a convenience plotting function to have a look at 
    the position, velocity, and acceleration data after initial 
    processing using extract_data. It is useful for sanity checks, 
    making sure everything looks reasonable. 

    Args:
       tracklist_subset (list): 
           List of strings indicating sequence names of desired trials.
       tracklist (dict):
           tracklist produced by extract_data()

    Returns
        None. Just plots.
          """

    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)
        data = tracklist[trial]['data']
        colors = plt.cm.cubehelix(scaled_time)
        m = cm.ScalarMappable(cmap=cm.cubehelix)
        m.set_array(tracklist[trial]['data'].index.values)

        # filename = str(trial) + '.pdf'

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist[trial]['behavior'])

        ax1 = fig.add_subplot(4, 2, 1, projection='3d')
        ax1.set_title('Pt 1 Position')
        ax1.scatter3D(xs=tracklist[trial]['data']['pt1x_smth'],
                      ys=tracklist[trial]['data']['pt1y_smth'],
                      zs=tracklist[trial]['data']['pt1z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax1.autoscale(enable=True, tight=True)
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        ax2 = fig.add_subplot(4, 2, 3)
        ax2.plot(data.index.values, data['pt1y_v_smth'], 'bo')
        ax2.set_ylabel('Y velocity (cm/s)', color='b')
        ax2.tick_params('y', colors='b')

        ax3 = fig.add_subplot(4, 2, 5)
        ax3.plot(data.index.values, data['pt1_net_v'], 'bo')
        ax3.set_ylabel('Net Velocity (cm/s)', color='b')
        ax3.tick_params('y', colors='b')

        ax4 = fig.add_subplot(4, 2, 7)
        ax4.plot(data.index.values, data['pt1_net_a'], 'bo')
        ax4.set_ylabel('Net accel (cm/s2)', color='b')
        ax4.tick_params('y', colors='b')

        ax5 = fig.add_subplot(4, 2, 2, projection='3d')
        ax5.set_title('Pt 2 Position')
        ax5.scatter3D(xs=tracklist[trial]['data']['pt2x_smth'],
                      ys=tracklist[trial]['data']['pt2y_smth'],
                      zs=tracklist[trial]['data']['pt2z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax5.autoscale(enable=True, tight=True)
        ax5.set_xlabel('X position')
        ax5.set_ylabel('Y position')
        ax5.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        ax6 = fig.add_subplot(4, 2, 4)
        ax6.plot(data.index.values, data['pt2y_v_smth'], 'bo')
        ax6.set_ylabel('Y velocity (cm/s)', color='b')
        ax6.tick_params('y', colors='b')

        ax7 = fig.add_subplot(4, 2, 6)
        ax7.plot(data.index.values, data['pt2_net_v'], 'bo')
        ax7.set_ylabel('Net Velocity (cm/s)', color='b')
        ax7.tick_params('y', colors='b')

        ax8 = fig.add_subplot(4, 2, 8)
        ax8.plot(data.index.values, data['pt2_net_a'], 'bo')
        ax8.set_ylabel('Net accel (cm/s2)', color='b')
        ax8.tick_params('y', colors='b')

        plt.show()