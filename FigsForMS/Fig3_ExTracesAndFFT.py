matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('mystyle.mplstyle')

# Fourier transforms with represetative traces
fig3 = plt.figure(figsize = (15,9))

# Bass Steady De-trended trace, 2Ls
ax1_3 = plot_trace_forfig(4, 3, 1, 'Bass1S11', tracklist)
ax1_3.set_title('Bass Steady Swimming')
# Bass LinAcc De-trended trace,2Ls
ax2_3 = plot_trace_forfig(4, 3, 2, 'Bass1S08', tracklist)
ax2_3.set_title('Bass Linear Acceleration')
# Bass Burst De-trended trace, 2Ls
ax3_3 = plot_trace_forfig(4, 3, 3, 'Bass1S13', tracklist)
ax3_3.set_title('Bass Burst Acceleration')
# Bass Steady FFT
ax4_3 = plot_fft_forfig(4, 3, 4, 'Bass1S11', tracklist)
# Bass LinAcc FFT
ax5_3 = plot_fft_forfig(4, 3, 5, 'Bass1S08', tracklist)
# Bass Burst FFT
ax6_3 = plot_fft_forfig(4, 3, 6, 'Bass1S13', tracklist)
# Trout Steady De-trended trace, 2Ls
ax7_3 = plot_trace_forfig(4, 3, 7, 'BTrout1S03', tracklist)
ax7_3.set_title('Trout Steady Swimming')
# Trout LinAcc De-trended trace, 2Ls
ax8_3 = plot_trace_forfig(4, 3, 8, 'BTrout2S01', tracklist)
ax8_3.set_title('Trout Linear Acceleration')
# Trout Burst De-trended trace, 3Ls
ax9_3 = plot_trace_forfig(4, 3, 9, 'BTrout2S03', tracklist)
ax9_3.set_title('Trout Burst Acceleration (3 BL/s)')
# Trout Steady FFT, 2Ls
ax10_3 = plot_fft_forfig(4, 3, 10, 'BTrout1S03', tracklist)
# Trout LinAcc FFT, 2Ls
ax11_3 = plot_fft_forfig(4, 3, 11, 'BTrout2S01', tracklist)
 # Trout Burst FFT, 3Ls
ax12_3 = plot_fft_forfig(4, 3, 12, 'BTrout2S03', tracklist)

plt.tight_layout()
plt.savefig('Fig3_TailTracesFFTS_2Ls.svg', fmt='svg')
plt.savefig('Fig3_TailTracesFFTS_2Ls.pdf', fmt='pdf')
plt.show()