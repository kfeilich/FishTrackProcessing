# FishTrackProcessing

## Synopsis

This is a module written to display and analyze 3D position data of
swimming fishes, with an emphasis on understanding transient changes
in caudal finbeat parameters and unsteady locomotion. This assumes
that you have digitized position data using DLT (Hedrick, 2008) saved in
flat format in a specific directory with a specified file naming
convention, with two points tracked from a lateral view video and a
ventral view video, with pt 1 being the tip of the snout and pt2
being the tip of the caudal fin. It also assumes that you have a
trial_info.csv file formatted as the one in my example data. You may
need to modify the code depending on your species/needs, but I tried
to make that obvious.

## Code example

```python
from extract_data import extract_data
from finbeat_calc import finbeat_calc
from store_data import store_data
from read_data import read_data
from make_subset import make_subset
from plot_accel import plot_accel
from plot_track import plot_track
from check_plots import check_plots
from plot_analysis import plot_analysis
from sin_corr import sin_corr
from plot_fit_sine import plot_fit_sine

# Take position data from hardcoded folder, calc. velocities, accels
tracklist = extract_data()

# Calculate finbeat periods, amplitudes, and subsequent accel
finbeats, finbeat_byP, finbeat_byT = finbeat_calc(tracklist.keys(),
                                             tracklist)
# Fit sine waves to tailbeat motion
sine_estimates = sin_corr(tracklist.keys, tracklist, finbeat_byP)
```

## Motivation

My dissertation research required an approach to swimming kinematics
that could look at both finbeat-specific performance and temporal
heterogeneity in kinematics.

## Installation

This is written for Python 3, and has the following dependencies:


* matplotlib
* peakutils: https://bitbucket.org/lucashnegri/peakutils
* numpy
* scipy
* tkinter
* pickle
* pandas
* statsmodels


Most of these are included in the Anaconda distribution of Python,
which I recommend. For those that aren't, see the links for
installation instructions.
Once those are installed clone this repository to your directory of
choice, initiate a jupyter notebook in that directory, and run the
code in a jupyter notebook. It's just easier that way. Feel free to
email me if you have difficulty.

## Reference

Documentation is ongoing, but most of the essential plotting and data
wrangling functions are well documented and commented within the
scripts. **When in doubt, run the script in process_points.py -- that's
the master scaffold for use of all the other functions.**


## Contributors

99% Kara Feilich ( @kfeilich ), 1% Ryan Feather ( @ryanfeather )

## License

Haven't decided yet.