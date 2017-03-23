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
speed = np.linspace(0.1,150, num=number_pts)
period = -0.0066*speed + 0.999934
amplitude = -0.217147 * np.log((0.00001)*((149/speed)-1))
effort = np.divide(amplitude,period)

# Set up diagnostic figure
fig1 = plt.figure(figsize = (24, 5))
# Subplot1: Speed by Period
ax1 = fig1.add_subplot(1,4,1)
ax1.set_xlabel('Caudal Finbeat Period (s)')
ax1.set_ylabel('Speed (cm/s)')
ax1.set_xlim(0,1)
ax1.set_ylim(0,speed_max)
ax1.plot(period, speed)

# Subplot2: Speed by Amplitude
ax2 = fig1.add_subplot(1,4,2)
ax2.set_xlabel('Caudal Finbeat Amplitude (cm)')
ax2.set_ylabel('Speed (cm/s)')
ax2.set_xlim(0,5)
ax2.set_ylim(0,speed_max)
ax2.plot(amplitude, speed)

# Subplot 3: Gait Space (x: Caudal Fin "Effort", y: Speed)
ax3 = fig1.add_subplot(1, 4, 3)
ax3.set_xlabel('Caudal Finbeat Effort (cm/s)')
ax3.set_ylabel('Speed (cm/s)')
ax3.set_xlim(0,20)
ax3.set_ylim(0,speed_max)
ax3.plot(effort, speed)

ax4 = fig1.add_subplot(1,4,4, projection='3d')
ax4.set_xlabel('Caudal Finbeat Period (s)')
ax4.set_ylabel('Caudal Finbeat Amplitude (cm)')
ax4.set_zlabel('Speed (cm/s)')
ax4.plot(period, amplitude, speed)

plt.savefig("Fig1a_PerAmpEffortSpeed.pdf", fmt='pdf', bbox_inches='tight')
plt.show()

fig1a =  plt.figure(figsize = (9, 4))
# Subplot 1: Gait Space (x: Caudal Fin "Effort", y: Speed)
ax5 = fig1a.add_subplot(1, 2, 1)
ax5.set_xlabel('Caudal Finbeat Effort (cm/s)')
ax5.set_ylabel('Speed (cm/s)')
ax5.set_xlim(0,20)
ax5.set_ylim(0,speed_max)
ax5.plot(effort,speed)

ax6 = fig1a.add_subplot(1,2,2, projection='3d')
ax6.set_xlabel('Caudal Finbeat Period (s)')
ax6.set_ylabel('Caudal Finbeat Amplitude (cm)')
ax6.set_zlabel('Speed (cm/s)')
ax6.set_xlim(0,1)
ax6.set_ylim(0,5)
ax6.plot(period, amplitude, speed)
plt.tight_layout()
plt.savefig("Fig1_EffortSpeed.pdf", fmt='pdf', bbox_inches='tight')
plt.show()
