# Load all of the things
import os
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rc('axes.formatter', useoffset=False)

def plot_tracks(tracklist_subset, tracklist):
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
        # Pull from colormap (here cubehelix)
        colors = plt.cm.cubehelix(scaled_time)

        # Raw Plot Pt 1, 3D
        fig = plt.figure()
        fig.suptitle(tracklist[trial]['sequence'])
        ax = fig.add_subplot(4, 2, 1, projection='3d')
        ax.set_title('Pt 1 Raw Position')
        p = ax.scatter3D(xs=tracklist[trial]['data']['pt1x'],
                         ys=tracklist[trial]['data']['pt1y'],
                         zs=tracklist[trial]['data']['pt1z'],
                         zdir='z', s=3, c=colors, marker='o',
                         edgecolor='none')  # 3D Scatter plot
        # ax.set_xlim3d(0,1000)
        # ax.set_ylim3d(0,1000)
        # ax.set_zlim3d(0,1000)
        ax.autoscale(enable=True, tight=True)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        m = cm.ScalarMappable(cmap=cm.cubehelix)
        m.set_array(tracklist[trial]['data'].index.values)
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Raw Plot Pt 2, 3D
        ax = fig.add_subplot(4, 2, 2, projection='3d')
        ax.set_title('Pt 2 Raw Position')
        p2 = ax.scatter3D(xs=tracklist[trial]['data']['pt2x'],
                          ys=tracklist[trial]['data']['pt2y'],
                          zs=tracklist[trial]['data']['pt2z'],
                          zdir='z', s=3, c=colors, marker='o',
                          edgecolor='none')  # 3D Scatter plot
        # ax2.set_xlim3d(0,1000)
        # ax2.set_ylim3d(0,1000)
        # ax2.set_zlim3d(0,1000)
        ax.autoscale(enable=True, tight=True)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Smoothed Data Pt 1, 3D
        ax = fig.add_subplot(4, 2, 3, projection='3d')
        ax.set_title('Pt 1 Smoothed Position')
        p3 = ax.scatter3D(xs=tracklist[trial]['data']['pt1x_smth'],
                          ys=tracklist[trial]['data']['pt1y_smth'],
                          zs=tracklist[trial]['data']['pt1z_smth'],
                          zdir='z', s=3, c=colors, marker='o',
                          edgecolor='none')  # Scatter plot
        # ax3.set_xlim3d(0,1000)
        # ax3.set_ylim3d(0,1000)
        # ax3.set_zlim3d(0,1000)
        ax.autoscale(enable=True, tight=True)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Smoothed Data Pt 2, 3D
        ax = fig.add_subplot(4, 2, 4, projection='3d')
        ax.set_title('Pt 2 Smoothed Position')
        p4 = ax.scatter3D(xs=tracklist[trial]['data']['pt2x_smth'],
                          ys=tracklist[trial]['data']['pt2y_smth'],
                          zs=tracklist[trial]['data']['pt2z_smth'],
                          zdir='z', s=3, c=colors, marker='o',
                          edgecolor='none')  # Scatter plot
        # #ax4.set_xlim3d(0,1000)
        # #ax4.set_ylim3d(0,1000)
        # #ax4.set_zlim3d(0,1000)
        ax.autoscale(enable=True, tight=True)
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Velocity Pt 1
        ax = fig.add_subplot(4, 2, 5)
        ax.set_title('Pt 1 Streamwise Velocity')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=-tracklist[trial]['data']['pt1x_v_smth'],
                    c=colors, edgecolor='none')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Velocity Pt 2
        ax = fig.add_subplot(4, 2, 6)
        ax.set_title('Pt 2 Streamwise Velocity')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=-tracklist[trial]['data']['pt2x_v_smth'],
                    c=colors, edgecolor='none')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Accel Pt 1
        ax = fig.add_subplot(4, 2, 7)
        ax.set_title('Pt 1 Streamwise Acceleration')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=-tracklist[trial]['data']['pt1x_a'],
                    c=colors, edgecolor='none')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Accel Pt 2
        ax = fig.add_subplot(4, 2, 8)
        ax.set_title('Pt 2 Streamwise Acceleration')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=-tracklist[trial]['data']['pt2x_a'],
                    c=colors, edgecolor='none')
        plt.colorbar(m, shrink=0.5, aspect=10)
        plt.show()
    '''