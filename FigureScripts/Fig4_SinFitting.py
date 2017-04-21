plt.style.use('mystyle.mplstyle')

fig4 = plt.figure(figsize = (15,5))


# Bass Steady De-trended trace, 2Ls
ax1_4 = plot_fit_sine_forfig(fig4, 2, 3, 1, 'Bass1S11', tracklist, corr_w_sin)
ax1_4.set_title('Bass Steady Swimming')
# Bass LinAcc De-trended trace,2Ls
ax2_4 = plot_fit_sine_forfig(fig4, 2, 3, 2, 'Bass1S08', tracklist, corr_w_sin)
ax2_4.set_title('Bass Linear Acceleration')
# Bass Burst De-trended trace, 2Ls
ax3_4 = plot_fit_sine_forfig(fig4, 2, 3, 3, 'Bass1S13', tracklist, corr_w_sin)
ax3_4.set_title('Bass Burst Acceleration')
# Trout Steady De-trended
ax4_4 = plot_fit_sine_forfig(fig4, 2, 3, 4, 'BTrout1S03', tracklist,
                             corr_w_sin)
ax4_4.set_title('Trout Steady Swimming')
# Trout LinAcc De-trended trace, 2Ls
ax5_4 = plot_fit_sine_forfig(fig4, 2, 3, 5, 'BTrout2S01', tracklist,
                             corr_w_sin)
ax5_4.set_title('Trout Linear Acceleration')
# Trout Burst De-trended trace, 3Ls
ax6_4 = plot_fit_sine_forfig(fig4, 2, 3, 6, 'BTrout2S03', tracklist,
                             corr_w_sin)
ax6_4.set_title('Trout Burst Acceleration (3 BL/s)')


plt.tight_layout()
#plt.savefig('Fig4_FitSines_Correls_2Ls.pdf', fmt='pdf')
#plt.savefig('Fig4_FitSines_Correls_2Ls.svg', fmt='svg')
plt.show()