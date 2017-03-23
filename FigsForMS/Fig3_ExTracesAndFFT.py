plt.style.use('mystyle.mplstyle')

# Fourier transforms with represetative traces
fig3 = plt.figure(figsize = (15,9))

ax1_3 = plot_trace_forfig(4, 3, 1)  # Bass Steady De-trended trace
ax2_3 = plot_trace_forfig(4, 3, 2)  # Bass LinAcc De-trended trace
ax3_3 = plot_trace_forfig(4, 3, 3)  # Bass Burst De-trended trace
ax4_3 = plot_fft_forfig(4, 3, 4)  # Bass Steady Fourier transform
ax5_3 = plot_fft_forfig(4, 3, 5)  # Bass LinAcc Fourier transform
ax6_3 = plot_fft_forfig(4, 3, 6)  # Bass Burst Fourier transform
ax7_3 = plot_trace_forfig(4, 3, 7)  # Trout Steady De-trended trace
ax8_3 = plot_trace_forfig(4, 3, 8)  # Trout LinAcc De-trended trace
ax9_3 = plot_trace_forfig(4, 3, 9)  # Trout Burst De-trended trace
ax10_3 = plot_fft_forfig(4, 3, 10)  # Trout Steady Fourier transform
ax11_3 = plot_fft_forfig(4, 3, 11)  # Trout LinAcc Fourier transform
ax12_3 =  plot_fft_forfig(4, 3, 12)  # Trout Burst Fourier transform