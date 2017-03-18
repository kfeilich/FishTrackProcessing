from extract_data import extract_data
from finbeat_calc import finbeat_calc
from store_data import store_data
from make_subset import make_subset
from plot_accel import plot_accel
from plot_track import plot_track
from check_plots import check_plots
from plot_analysis import plot_analysis

# Take position data from hardcoded folder, calc. velocities, accels
tracklist = extract_data()

# Calculate finbeat peaks and troughs
# Extract finbeat periods, amplitudes, and subsequent accel
finbeats, finbeat_byP, finbeat_byT = finbeat_calc(tracklist.keys(),
                                             tracklist)

#Pickle the data
store_data('data')

tracklist, finbeats, finbeat_byP, finbeat_byT = read_data('data.pickle')
# Plot things if desired
# make_subset(group_by, identifier, tracklist)
# plot_track(['sometrial'], tracklist)
# plot_accel(['sometrial'], tracklist)
# check_plots(['sometrial'], tracklist)
# plot_analysis('subset_name'),['sometrial'], finbeat_data,
# tracklist)

# """If you want to plot the peaks and troughs on the position data,
# uncomment section in finbeat_calc"""
