"""These are just some usage examples, but you may as well just look 
at the ones in the jupyter notebook. """

from extract_data import extract_data
from finbeat_calc import finbeat_calc
from read_data import read_data
from sin_corr import sin_corr
from store_data import store_data

# Take position data from hardcoded folder, calc. velocities, accels
tracklist = extract_data()

# Calculate finbeat peaks and troughs
# Extract finbeat periods, amplitudes, and subsequent accel
finbeats, finbeat_byP, finbeat_byT = finbeat_calc(tracklist.keys(),
                                             tracklist)

# Save out the data for later
store_data('data')

# Read stored data
tracklist, finbeats, finbeat_byP, finbeat_byT = read_data('data.pickle')

sine_estimates = sin_corr(tracklist.keys, tracklist, finbeat_byP)
# Plot things if desired

# make_subset(group_by, identifier, tracklist)
# plot_track(['sometrial'], tracklist)
# plot_accel(['sometrial'], tracklist)
# check_plots(['sometrial'], tracklist)
# plot_analysis('subset_name'),['sometrial'], finbeat_data,
# tracklist)
# plot_fit_sine(['sometrial'], tracklist, sine_estimates)

# """If you want to plot the peaks and troughs on the position data,
# uncomment section in finbeat_calc"""