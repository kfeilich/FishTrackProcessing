# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:24:53 2017

@author: Kara
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sympy import *
from scipy.signal import savgol_filter


#import points
data = np.genfromtxt("E:/Digitized tracks/BTrout1_S08_tracks_xyzmerge.csv",delimiter=',',skip_header=1)


#extract useful variables

pt1x = data[:,1] #point 1 x-component
pt1y = data[:,2] #point 1 y-component
pt1z = data[:,0] #point 1 z-component

pt2x = data[:,4] #point 2 x-component
pt2y = data[:,5] #point 2 y-component
pt2z = data[:,3] #point 2 z-component

#turn positions into cm from pixels


#generate time array
time = np.linspace(0,len(pt1x)*0.002,num=len(pt1x),endpoint=False)
dt = 0.002

#plot Raw 3d position, color by time
scaled_time = (time-time.min())/time.ptp() #scale time for colormap
colors = plt.cm.cubehelix(scaled_time) #pull from colormap (here cubehelix)

#Raw Plot Pt 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(xs=pt1x, ys=pt1y, zs=pt1z,zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax.set_xlim3d(0,1000)
#ax.set_ylim3d(0,1000)
#ax.set_zlim3d(0,1000)
ax.autoscale(enable=True, tight=None)
plt.show()

#Raw Plot Pt 2
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter3D(xs=pt2x, ys=pt2y, zs=pt2z, zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax2.set_xlim3d(0,1000)
#ax2.set_ylim3d(0,1000)
#ax2.set_zlim3d(0,1000)
ax2.autoscale(enable=True, tight=None)
plt.show()


#smooth data using savitzky golay
pt1x_smth = scipy.signal.savgol_filter(pt1x, window_length = 91, polyorder = 2)
pt1y_smth = scipy.signal.savgol_filter(pt1y, window_length = 91, polyorder = 2)
pt1z_smth = scipy.signal.savgol_filter(pt1z, window_length = 91, polyorder = 2)

pt2x_smth = scipy.signal.savgol_filter(pt2x, window_length = 91, polyorder = 2)
pt2y_smth = scipy.signal.savgol_filter(pt2y, window_length = 91, polyorder = 2)
pt2z_smth = scipy.signal.savgol_filter(pt2z, window_length = 91, polyorder = 2)


#plot pt1 smooth data 
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter3D(xs=pt1x_smth, ys=pt1y_smth, zs=pt1z_smth,zdir='z', s=3,c=colors, marker='o', edgecolor='none')#Scatter plot
#ax3.set_xlim3d(0,1000)
#ax3.set_ylim3d(0,1000)
#ax3.set_zlim3d(0,1000)
ax3.autoscale(enable=True, tight=None)
plt.show()

#plot pt2 smooth data 
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

#Smooth Vel
pt1x_v_smth = scipy.signal.savgol_filter(pt1x_v, window_length = 51, polyorder = 2)
pt1y_v_smth = scipy.signal.savgol_filter(pt1y_v, window_length = 51, polyorder = 2)
pt1z_v_smth = scipy.signal.savgol_filter(pt1z_v, window_length = 51, polyorder = 2)

pt2x_v_smth = scipy.signal.savgol_filter(pt2x_v, window_length = 51, polyorder = 2)
pt2y_v_smth = scipy.signal.savgol_filter(pt2y_v, window_length = 51, polyorder = 2)
pt2z_v_smth = scipy.signal.savgol_filter(pt2z_v, window_length = 51, polyorder = 2)

#DERIVATIVES -- ACCEL
pt1x_a = np.ediff1d(pt1x_v_smth)
pt1y_a = np.ediff1d(pt1y_v_smth)
pt1z_a = np.ediff1d(pt1z_v_smth)

pt2x_a = np.ediff1d(pt2x_v_smth)
pt2y_a = np.ediff1d(pt2y_v_smth)
pt2z_a = np.ediff1d(pt2z_v_smth)

time_a = time_v[:len(time_v)-1]


plt.scatter(y=-pt1x_a, x = time_a, )
plt.show()
