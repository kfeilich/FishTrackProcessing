def plot_accels_forfig(plotnum, trial, tracklist):
    """Plot tail beat acceleration and snout streamwise acceleration 
       This function is used within a figure script to produce panels 
       for a composite figure. See "Fig6_Trial Traj.py"
       Does what it says above, using data from tracklist. Tail tip 
       accelerations are presented as absolute values. 

               Args:
                   plotnum (int): the number indicating the position 
                   of the panel using GridSpec conventions
                   trial(str): a trial name
                   tracklist (dict): tracklist produced by 
                        extract_data()
                
                Returns: 
                    ax, ax2 (matplotlib axes): the panel showing 
                    acceleration data

              """
    ax = fig6.add_subplot(plotnum)
    ax.plot(tracklist[trial]['data'].index.values,
                abs(tracklist[trial]['data']['pt2_net_a']),c='blue')
    ax2 = ax.twinx()
    ax2.plot(tracklist[trial]['data'].index.values,
                tracklist[trial]['data']['pt1_net_a'], c='orange')
    ax.set_ylim(tracklist[trial]['data']['pt1_net_a'].min(),
             abs(tracklist[trial]['data']['pt2_net_a'].max()))
    ax2.set_ylim(tracklist[trial]['data']['pt1_net_a'].min(),
            abs(tracklist[trial]['data']['pt1_net_a'].max()))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tailtip Acceleration (cm$^2$)')
    ax.yaxis.label.set_color('blue')
    ax2.set_ylabel('Snout Acceleration (cm$^2$)')
    ax2.yaxis.label.set_color('orange')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,2))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,2))
    return ax, ax2