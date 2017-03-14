from extract_data import extract_data
from finbeat_calc import finbeat_calc
from plot_accel import plot_accel
from plot_track import plot_track
from check_plots import check_plots

# Take position data from hardcoded folder, calc. velocities, accels
tracklist = extract_data()

# Calculate finbeat peaks and troughs

finbeats, finbeat_byP, finbeat_byT = finbeat_calc(tracklist.keys(),
                                             tracklist)

# Extract finbeat periods, amplitudes, and subsequent accel

# Plot things if desired
# plot_track(['sometrial'], tracklist)
# plot_accel(['sometrial'], tracklist)
# check_plots(['sometrial'], tracklist)

# """If you want to plot the peaks and troughs on the position data,
# uncomment section in finbeat_calc"""