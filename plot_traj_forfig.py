import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from make_subset import make_subset
from mpl_toolkits.mplot3d import Axes3D


def plot_traj_forfig(nrow, ncol, plotnum, trial, tracklist,
                     finbeat_subset):
    behavior = tracklist[trial]['behavior']
    species = tracklist[trial]['species']
    speed = tracklist[trial]['start_spd_BLs']

    steady_subset = make_subset(group_by1='species',
                                identifier1=species,
                                tracklist=tracklist,
                                group_by2='behavior',
                                identifier2='S',
                                group_by3='start_spd_BLs',
                                identifier3=speed)
    steady_finbeats = []
    # Make (speed, period, amplitude) tuples
    for i in steady_subset:
        for j in finbeat_subset[i].index.values:
            start = finbeat_subset[i]['time'][j]
            stop = finbeat_subset[i]['endtime'][j]
            speed = tracklist[i]['data'][
                        'pt1_net_v'][start:stop].max()
            steady_finbeats.append((speed,
                                    finbeat_subset[i]['period'][j],
                                    finbeat_subset[i]['amplitude'][j]))

    trial_finbeats = []
    count = 0
    for k in finbeat_subset[trial].index.values:
        start = finbeat_subset[trial]['time'][k]
        stop = finbeat_subset[trial]['endtime'][k]
        period = finbeat_subset[trial]['period'][k]
        amplitude = finbeat_subset[trial]['amplitude'][k]
        trial_finbeats.append((count, tracklist[trial]['data'][
                                          'pt1_net_v'][
                                      start:stop].max(), period,
                               amplitude))
        count += 1

    nans = {np.nan, float('nan')}
    trial_finbeats = [n for n in trial_finbeats if
                      not nans.intersection(n)]

    allz = []
    for i in steady_finbeats:
        allz.append(i[0])

    if allz == []:
        min_z = 0
    else:
        min_z = min(allz)

    ax1 = fig6.add_subplot(nrow, ncol, plotnum, projection='3d')
    # fig.suptitle(trial +' '+ behavior)
    for a, b, c in steady_finbeats:
        ax1.plot(xs=[b, b], ys=[c, c], zs=[min_z, a],
                 linestyle='solid', c='black',
                 alpha=0.8, linewidth=0.5)
        ax1.scatter3D(xs=b,
                      ys=c,
                      zs=a,
                      zdir='z', s=30, marker='o',
                      c='black',
                      edgecolor='none')
    for a, b, c, d in trial_finbeats:
        ax1.plot(xs=[c, c], ys=[d, d], zs=[min_z, b],
                 linestyle='solid', c='blue',
                 alpha=0.8, linewidth=0.5)
        ax1.scatter3D(xs=c,
                      ys=d,
                      zs=b,
                      zdir='z', s=40, marker='o',
                      c='blue',
                      edgecolor='none')
        ax1.text(x=c, y=d, z=b, s=str(a), color='red', fontsize=16)
    ax1.set_xlabel('Period(s)')
    ax1.set_ylabel('Amplitude(cm)')
    ax1.set_zlabel('Max. Inst. Speed (cm/s)')
    return ax1