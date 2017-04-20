def plot_trace_forfig(rows, columns, number, trial, tracklist):
    """Plots detrended tailtip motion against time.

        Does what it says above. Intended for use to create panels 
        for composite figures. See Figure 5 for example. 

        Args:
            rows (int): Number of rows in composite figure
            columns (int): Number columns in composite figure
            number (int): Number of this subplot
            trial(string): the trial name
            tracklist (dict): the compiled position, velocity,
                              and acceleration data for all trials
                               produced by extract_data()
          
        Returns:
            ax (matplotlib Axes)
        """

    raw_data = tracklist[trial]['data']['pt2y']
    behavior = tracklist[trial]['behavior']
    fish = tracklist[trial]['fish']
    init_speed = tracklist[trial]['start_spd_BLs']
    base = peakutils.baseline(raw_data, 3)  # Find bkgrd trend
    raw_data = raw_data - base
    time = raw_data.index.values

    if fish == 'Bass1':
        col = 'darkblue'
    else:
        col = 'darkred'

    ax = fig3.add_subplot(rows, columns, number)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tail Tip Position (cm)')
    ax.plot(time, raw_data, col)

    return ax