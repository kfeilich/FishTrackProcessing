import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# NOTE: This is not real data, just a schematic
plt.style.use('mystyle.mplstyle')

# Pull a colormap
cm = plt.get_cmap("plasma")

# Set axis and colorbar limits
effort_max = 100.0
freq_max = 10.0
amp_max = 5.0
speed_max = 150.0
number_pts = 100

# Simulate some data
effort = np.linspace(start=0.0,stop=effort_max, num=number_pts)
speed = np.linspace(start=0.0, stop=speed_max, num=number_pts)
period = np.linspace(start=1./freq_max, stop=1, num=number_pts)
amplitude = effort*period

# Set up the figure
fig1 = plt.figure(figsize = (12, 5))

# Subplot 1: Gait Space (x: Caudal Fin "Effort", y: Speed)
ax1 = fig1.add_subplot(1, 2, 1)
ax1.set_xlabel('Caudal Fin Effort (cm/s)')
ax1.set_ylabel('Speed (cm/s)')
ax1.set_xlim3d(0, effort_max)
ax1.set_ylim3d(0, speed_max)
ax1.plot(x=effort, y=speed)

# Subplot 2: Gait Space broken out by period and amplitude
ax2 = fig1.add_subplot(1, 2, 2, projection = '3d')
ax2.set_xlabel('Period (cm/s)')
ax2.set_ylabel('Amplitude (cm/s)')
ax2.set_zlabel('Speed (cm/s)')
ax2.set_xlim3d(0, effort_max)
ax2.set_ylim3d(0, )
ax2.set_zlim(0,speed_max)
ax2.plot(x=period, y=amplitude, z=speed)
