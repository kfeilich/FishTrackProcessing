import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# NOTE: This is not real data, just a schematic
plt.style.use('mystyle.mplstyle')

# Pull a colormap
cm = plt.get_cmap("plasma")

# Set axis and colorbar limits
effort_max = 100.0
amp_max = 5.0
speed_max = 150.0
number_pts = 100
per_max = 1.0

# Simulate some data
y = np.linspace(0.0,150, num=number_pts)
x_period = np.linspace(start=0.01, stop=per_max, num=number_pts)
period = (-149.0*x_period) + 150.0
x_frequency = np.linspace(start = 0.01, stop = 10, num = number_pts)
x_amplitude = np.linspace(start=0.01, stop=amp_max, num=100)
amplitude = 150.0/(1.0+10.0**((2.5-x_amplitude)*3.0))  # arbitrary but approx. sigmoid
amp_by_per = 5./2. - ((np.log(150/(150-149*x_period)))**(1./3.))/((np.log(10))**(1/3))

# Set up the figure
fig1 = plt.figure(figsize = (24, 5))
ax1 = fig1.add_subplot(1,4,1)
ax1.set_xlabel('Caudal Finbeat Period (s)')
ax1.set_ylabel('Speed (cm/s)')
ax1.set_xlim(0,1)
ax1.set_ylim(0,speed_max)
ax1.plot(x_period, period)

ax2 = fig1.add_subplot(1,4,2)
ax2.set_xlabel('Caudal Finbeat Amplitude (cm)')
ax2.set_ylabel('Speed (cm/s)')
#ax2.set_xlim(0,)
ax2.set_ylim(0,speed_max)
ax2.plot(x_amplitude, amplitude)

# Subplot 1: Gait Space (x: Caudal Fin "Effort", y: Speed)
ax3 = fig1.add_subplot(1, 4, 3)
ax3.set_xlabel('Caudal Finbeat Effort (cm/s)')
ax3.set_ylabel('Speed (cm/s)')
#ax1.set_xlim(0,)
ax3.set_ylim(0,speed_max)
ax3.plot(x_period*x_frequency, y)

ax4 = fig1.add_subplot(1,4,4, projection='3d')
ax4.set_xlabel('Caudal Finbeat Period (s)')
ax4.set_ylabel('Caudal Finbeat Amplitude (cm)')
ax4.set_zlabel('Speed (cm/s)')
ax4.plot(x_period, amp_by_per, period)

## Subplot 2: Gait Space broken out by period and amplitude
#ax2 = fig1.add_subplot(1, 2, 2, projection = '3d')
#ax2.set_xlabel('Period (cm/s)')
#ax2.set_ylabel('Amplitude (cm/s)')
#ax2.set_zlabel('Speed (cm/s)')
#ax2.set_xlim3d(0, per_max)
#ax2.set_ylim3d(0, amp_max)
#ax2.set_zlim(0,speed_max)
#ax2.plot(xs=x_period, ys=x_amplitude, zs=amplitude)
plt.show()

fig1a =  plt.figure(figsize = (12, 5))
# Subplot 1: Gait Space (x: Caudal Fin "Effort", y: Speed)
ax5 = fig1a.add_subplot(1, 2, 1)
ax5.set_xlabel('Caudal Finbeat Effort (cm/s)')
ax5.set_ylabel('Speed (cm/s)')
#ax5.set_xlim(0,)
ax5.set_ylim(0,speed_max)
ax5.plot(x_period*x_frequency, y)

ax6 = fig1a.add_subplot(1,2,2, projection='3d')
ax6.set_xlabel('Caudal Finbeat Period (s)')
ax6.set_ylabel('Caudal Finbeat Amplitude (cm)')
ax6.set_zlabel('Speed (cm/s)')
ax6.plot(x_period, amp_by_per, period)
plt.show()
