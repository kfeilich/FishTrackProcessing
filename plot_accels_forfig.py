import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from make_subset import make_subset
from mpl_toolkits.mplot3d import Axes3D

def plot_accels_forfig(nrow, ncol, plotnum, trial, tracklist):

    ax = fig6.add_subplot(nrow, ncol, plotnum)
    ax.scatter(tracklist[trial]['data'].index.values,
                abs(tracklist[trial]['data']['pt2_net_a']),c='blue')
    ax2 = ax.twinx()
    ax2.scatter(tracklist[trial]['data'].index.values,
                tracklist[trial]['data']['pt1_net_a'], c='black')
    ax.ylim(tracklist[trial]['data']['pt1_net_a'].min(),
             abs(tracklist[trial]['data']['pt2_net_a'].max()))
    ax2.ylim(tracklist[trial]['data']['pt1_net_a'].min(),
            abs(tracklist[trial]['data']['pt1_net_a'].max()))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tailtip Acceleration (cm$^2$)')
    ax2.set_ylabel('Snout Acceleration (cm$^2$)')
    return ax, ax2