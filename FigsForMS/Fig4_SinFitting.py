plt.style.use('mystyle.mplstyle')

fig4 = plt.figure(figsize = (15,5))


# Bass Steady De-trended trace, 2Ls
ax1_4 = plot_fit_sine_forfig(2, 3, 1, 'Bass1S11', tracklist, corr_w_sin)
ax1_4.set_title('Bass Steady Swimming')
# Bass LinAcc De-trended trace,2Ls
ax2_4 = plot_fit_sine_forfig(2, 3, 2, 'Bass1S08', tracklist, corr_w_sin)
ax2_4.set_title('Bass Linear Acceleration')
# Bass Burst De-trended trace, 2Ls
ax3_4 = plot_fit_sine_forfig(2, 3, 3, 'Bass1S13', tracklist, corr_w_sin)
ax3_4.set_title('Bass Burst Acceleration')
# Trout Steady De-trended
ax7_3 = plot_fit_sine_forfig(2, 3, 4, 'BTrout1S03', tracklist, corr_w_sin)
ax7_3.set_title('Trout Steady Swimming')
# Trout LinAcc De-trended trace, 2Ls
ax8_3 = plot_fit_sine_forfig(2, 3, 5, 'BTrout2S01', tracklist, corr_w_sin)
ax8_3.set_title('Trout Linear Acceleration')
plt.tight_layout()

plt.savefig('Fig4_FitSines_Correls_2Ls.pdf', fmt='pdf')
plt.show()