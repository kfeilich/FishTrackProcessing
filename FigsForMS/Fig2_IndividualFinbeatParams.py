#import matplotlib
#import matplotlib.pyplot as plt
#from plot_analysis_forfig import plot_analysis_forfig

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.style.use('mystyle.mplstyle')



speeds_cb = [0]*len(tracklist.keys())
count_cb= 0
for i in tracklist.keys():
    speeds_cb[count_cb] = tracklist[i]['start_spd']
    count_cb+=1
speed_cb = max(speeds_cb)

fig2 = plt.figure(figsize = (12,15))

ax2_1 = plot_analysis_forfig(3,2,1, bass_subset, finbeat_byP, tracklist, 'A', True, False)  # Bass acc no cutoff
ax2_1.set_title('Bass \n')
ax2_2 = plot_analysis_forfig(3,2,2, trout_subset, finbeat_byP, tracklist, 'A', True, False)  # Trout accel no cutoff
ax2_2.set_title('Trout \n')
ax2_3 = plot_analysis_forfig(3,2,3, bass_subset, finbeat_byP, tracklist, 'A', True, True)  # Bass accel cutoff
ax2_4 = plot_analysis_forfig(3,2,4, trout_subset, finbeat_byP, tracklist, 'A', True, True)  # Trout accel cutoff
ax2_5 = plot_analysis_forfig(3,2,5, bass_subset, finbeat_byP, tracklist, 'V', True, False)  # Bass inst velocity
ax2_6 = plot_analysis_forfig(3,2,6, trout_subset, finbeat_byP, tracklist, 'V', True, False)  # Trout inst. velocity


fig2.text(0, 0.8, 'Acceleration\nAll Points Plotted\n ', va='center', rotation='vertical', fontsize=16, multialignment = 'center')
fig2.text(0, 0.49, 'Acceleration\nOnly Low Accelerations\n ', va='center', rotation='vertical', fontsize=16, multialignment = 'center')
fig2.text(0, 0.15, 'Velocity\nAll Points Plotted\n ', va='center', rotation='vertical', fontsize=16, multialignment = 'center')
#plt.show()
plt.tight_layout()
fig2.subplots_adjust(right=0.9)
cbar_ax = fig2.add_axes([0.93, 0.2, 0.03, 0.6])  # [left, bottom, width, height]
cbar_ax.set_xmargin(0.2)
cmap = matplotlib.cm.cool
norm = matplotlib.colors.Normalize(vmin=0, vmax=speed_cb)
cb1 = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label('Initial Speed (cm/s)')

#fig2.savefig('Fig2_IndividualFinbeats.pdf', fmt = 'pdf')
#fig2.savefig('Fig2_IndividualFinbeats.svg', fmt = 'svg')
plt.show()

