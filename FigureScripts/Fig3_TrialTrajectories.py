matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('mystyle.mplstyle')
import matplotlib.gridspec as gridspec

# Fourier transforms with represetative traces
fig3 = plt.figure(figsize = (14,8))
gs = gridspec.GridSpec(4, 3, height_ratios=[3,1,3,1])

# Bass Steady De-trended trace, 2Ls
ax1_3 = plot_traj_forfig(fig3,gs[0], 'Bass1S11', tracklist, finbeat_byP)
ax1_3.set_title('Bass Steady Swimming')
# Bass LinAcc De-trended trace,2Ls
ax2_3 = plot_traj_forfig(fig3,gs[1], 'Bass1S08', tracklist, finbeat_byP)
ax2_3.set_title('Bass Linear Acceleration')
# Bass Burst De-trended trace, 2Ls
ax3_3 = plot_traj_forfig(fig3,gs[2], 'Bass1S13', tracklist, finbeat_byP)
ax3_3.set_title('Bass Burst Acceleration')
# Bass Steady FFT
ax4_3 = plot_accels_forfig(fig3, gs[3], 'Bass1S11', tracklist)
# Bass LinAcc FFT
ax5_3 = plot_accels_forfig(fig3, gs[4], 'Bass1S08', tracklist)
# Bass Burst FFT
ax6_3 = plot_accels_forfig(fig3, gs[5], 'Bass1S13', tracklist)
# Trout Steady De-trended trace, 2Ls
ax7_3 = plot_traj_forfig(fig3, gs[6], 'BTrout1S03', tracklist, finbeat_byP)
ax7_3.set_title('Trout Steady Swimming')
# Trout LinAcc De-trended trace, 2Ls
ax8_3 = plot_traj_forfig(fig3, gs[7], 'BTrout2S01', tracklist, finbeat_byP)
ax8_3.set_title('Trout Linear Acceleration')
# Trout Burst De-trended trace, 3Ls
ax9_3 = plot_traj_forfig(fig3, gs[8], 'BTrout2S03', tracklist, finbeat_byP)
ax9_3.set_title('Trout Burst Acceleration (3 BL/s)')
# Trout Steady FFT, 2Ls
ax10_3 = plot_accels_forfig(fig3, gs[9], 'BTrout1S03', tracklist)
# Trout LinAcc FFT, 2Ls
ax11_3 = plot_accels_forfig(fig3, gs[10], 'BTrout2S01', tracklist)
 # Trout Burst FFT, 3Ls
ax12_3 = plot_accels_forfig(fig3, gs[11], 'BTrout2S03', tracklist)

plt.tight_layout()
# plt.savefig('Fig3_Finbeats_2Ls.svg', fmt='svg')
# plt.savefig('Fig3_Finbeats_2Ls.pdf', fmt='pdf')
plt.show()