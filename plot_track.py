# Load all of the things
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rc('axes.formatter', useoffset=False)


def plot_track(tracklist_subset, tracklist):
    # TODO
    """Plot the 3D position, streamwise velocity, and streamwise accel of the
        snout and the tail tip from data as produced
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

    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)

        # Determining axis limits
        pt1max_v = tracklist[trial]['data']['pt1x_v_smth'].max()
        pt1min_v = tracklist[trial]['data']['pt1x_v_smth'].min()
        pt1max_a = tracklist[trial]['data']['pt1x_a'].max()
        pt1min_a = tracklist[trial]['data']['pt1x_a'].min()
        pt1_vbuff = (pt1max_v - pt1min_v) * 0.05  # Adds margin of 5%
        pt1_abuff = (pt1max_a - pt1min_a) * 0.05
        pt1_vmaxlim = pt1max_v + pt1_vbuff
        pt1_vminlim = pt1min_v - pt1_vbuff
        pt1_amaxlim = pt1max_a + pt1_abuff
        pt1_aminlim = pt1min_a - pt1_abuff

        pt2max_v = tracklist[trial]['data']['pt2x_v_smth'].max()
        pt2min_v = tracklist[trial]['data']['pt2x_v_smth'].min()
        pt2max_a = tracklist[trial]['data']['pt2x_a'].max()
        pt2min_a = tracklist[trial]['data']['pt2x_a'].min()
        pt2_vbuff = (pt2max_v - pt2min_v) * 0.05
        pt2_abuff = (pt2max_a - pt2min_a) * 0.05
        pt2_vmaxlim = pt2max_v + pt2_vbuff
        pt2_vminlim = pt2min_v - pt2_vbuff
        pt2_amaxlim = pt2max_a + pt2_abuff
        pt2_aminlim = pt2min_a - pt2_abuff

        # Pull from colormap (here cubehelix)
        colors = plt.cm.cubehelix(scaled_time)

        # Raw Plot Pt 1, 3D
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(15)
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist['trial']['behavior'])
        ax1 = fig.add_subplot(4, 2, 1, projection='3d')
        ax1.set_title('Pt 1 Raw Position')
        ax1.scatter3D(xs=tracklist[trial]['data']['pt1x'],
                      ys=tracklist[trial]['data']['pt1y'],
                      zs=tracklist[trial]['data']['pt1z'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax1.autoscale(enable=True, tight=True)
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_zlabel('Z position')
        m = cm.ScalarMappable(cmap=cm.cubehelix)
        m.set_array(tracklist[trial]['data'].index.values)
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Raw Plot Pt 2, 3D
        ax2 = fig.add_subplot(4, 2, 2, projection='3d')
        ax2.set_title('Pt 2 Raw Position')
        ax2.scatter3D(xs=tracklist[trial]['data']['pt2x'],
                      ys=tracklist[trial]['data']['pt2y'],
                      zs=tracklist[trial]['data']['pt2z'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax2.autoscale(enable=True, tight=True)
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Y position')
        ax2.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Smoothed Data Pt 1, 3D
        ax3 = fig.add_subplot(4, 2, 3, projection='3d')
        ax3.set_title('Pt 1 Smoothed Position')
        ax3.scatter3D(xs=tracklist[trial]['data']['pt1x_smth'],
                      ys=tracklist[trial]['data']['pt1y_smth'],
                      zs=tracklist[trial]['data']['pt1z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # Scatter plot
        ax3.autoscale(enable=True, tight=True)
        ax3.set_xlabel('X position')
        ax3.set_ylabel('Y position')
        ax3.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Smoothed Data Pt 2, 3D
        ax4 = fig.add_subplot(4, 2, 4, projection='3d')
        ax4.set_title('Pt 2 Smoothed Position')
        ax4.scatter3D(xs=tracklist[trial]['data']['pt2x_smth'],
                      ys=tracklist[trial]['data']['pt2y_smth'],
                      zs=tracklist[trial]['data']['pt2z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # Scatter plot
        ax4.autoscale(enable=True, tight=True)
        ax4.set_xlabel('X position')
        ax4.set_ylabel('Y position')
        ax4.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Velocity Pt 1
        ax5 = fig.add_subplot(4, 2, 5)
        ax5.set_title('Pt 1 Streamwise Velocity')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt1x_v_smth'],
                    c=colors, edgecolor='none')
        ax5.set_xlim([0, timemax])
        ax5.set_ylim([pt1_vminlim, pt1_vmaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Velocity Pt 2
        ax6 = fig.add_subplot(4, 2, 6)
        ax6.set_title('Pt 2 Streamwise Velocity')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt2x_v_smth'],
                    c=colors, edgecolor='none')
        ax6.set_xlim([0, timemax])
        ax6.set_ylim([pt2_vminlim, pt2_vmaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Accel Pt 1
        ax7 = fig.add_subplot(4, 2, 7)
        ax7.set_title('Pt 1 Streamwise Acceleration')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt1x_a'],
                    c=colors, edgecolor='none')
        ax7.set_xlim([0, timemax])
        ax7.set_ylim([pt1_aminlim, pt1_amaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Accel Pt 2
        ax8 = fig.add_subplot(4, 2, 8)
        ax8.set_title('Pt 2 Streamwise Acceleration')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt2x_a'],
                    c=colors, edgecolor='none')
        ax8.set_xlim([0, timemax])
        ax8.set_ylim([pt2_aminlim, pt2_amaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)
        plt.show()