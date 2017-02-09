# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:24:53 2017

@author: Kara
"""
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy



scaled_time = (data.index.values - data.index.values.min()) / data.index.ptp()  # scale time for colormap
colors = plt.cm.cubehelix(scaled_time)  # pull from colormap (here cubehelix)

#Raw Plot Pt 1, 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(xs=pt1x, ys=pt1y, zs=pt1z,zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax.set_xlim3d(0,1000)
#ax.set_ylim3d(0,1000)
#ax.set_zlim3d(0,1000)
ax.autoscale(enable=True, tight=None)
fig.colorbar(cubehelix, shrink=0.5, aspect=10)
plt.show()

#Raw Plot Pt 2, 3D
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter3D(xs=pt2x, ys=pt2y, zs=pt2z, zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax2.set_xlim3d(0,1000)
#ax2.set_ylim3d(0,1000)
#ax2.set_zlim3d(0,1000)
ax2.autoscale(enable=True, tight=None)
plt.show()

#Smoothed Data Pt 1, 3D
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter3D(xs=pt1x_smth, ys=pt1y_smth, zs=pt1z_smth,zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax3.set_xlim3d(0,1000)
#ax3.set_ylim3d(0,1000)
#ax3.set_zlim3d(0,1000)
ax3.autoscale(enable=True, tight=None)
plt.show()

#Smoothed Dara Pt 2, 3D
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter3D(xs=pt2x_smth, ys=pt2y_smth, zs=pt2z_smth,zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax4.set_xlim3d(0,1000)
#ax4.set_ylim3d(0,1000)
#ax4.set_zlim3d(0,1000)
ax4.autoscale(enable=True, tight=None)
plt.show()

#DERIVATIVES -- VELOCITY
pt1x_v = np.ediff1d(pt1x_smth)
pt1y_v = np.ediff1d(pt1y_smth)
pt1z_v = np.ediff1d(pt1z_smth)

pt2x_v = np.ediff1d(pt2x_smth)
pt2y_v = np.ediff1d(pt2y_smth)
pt2z_v = np.ediff1d(pt2z_smth)

time_v = time[:len(time)-1]

#PLOT WHATEVS
plt.scatter(x = time_v, y=-pt1x_v)
plt.show()

time_a = time_v[:len(time_v)-1]


plt.scatter(y=-pt1x_a, x = time_a, )
plt.show()
