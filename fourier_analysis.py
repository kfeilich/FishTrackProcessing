import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import peakutils
matplotlib.rc('axes.formatter', useoffset=False)


def fourier_analysis(tracklist_subset, tracklist):
    """Computes full width at half maximum from FFT of tailbeat movement

        This function takes the trial information and data for a
        subset of the trials in a tracklist compiled by extract_data(),
        and finds the fin beat peaks and troughs in the tail (pt 2)
        position data. Using the module peakutils, this function
        prepares the position data for peak finding by fitting and
        removing a linear baseline for each trial, and produces indices
        and values for the peaks and troughs. It returns 3 dictionaries,
        one containing the timing, magnitude, and type of the local extrema
        for each trial; one containing finbeat start and end times,
        periods, and amplitudes as measured from the peak; and anther
        doing the same, but measuring from the troughs.

        Args:
            tracklist_subset (list): a list of strings with the trial
                                names of the desired trials from tracklist.
                                 Note: The list (even of a single
                                 element) must be contained in square
                                 brackets.
                                 Also note: Alternatively, to iterate
                                 over all trials, set this to
                                 tracklist.keys()
            tracklist (dict): a tracklist produced by extract_data()
        Returns:


"""

for trial in tracklist_subset:  # Iterate over desired trials
    trial_name = tracklist[trial]['sequence']
    # print(trial_name)  # for diagnostics
    fish = tracklist[trial]['fish']
    tailtip = tracklist[trial]['data']['pt2y_smth']
    time = tracklist[trial]['data'].index.values
    base = peakutils.baseline(tailtip, 1)  # Find linear bkgrd trend
    framerate = tracklist[trial]['FPS']
    behavior = tracklist[trial]['behavior']

    # Find best guess for number of finbeats using FFT
    fourier = np.fft.fft(tailtip)
    frequencies = np.fft.fftfreq(len(time), 1 / framerate)
    positive_frequencies = frequencies[np.where(frequencies >= 0)]
    magnitudes = abs(fourier[np.where(frequencies >= 0)])
    peak_frequency = np.argmax(magnitudes)
    guess = max(
        time) * peak_frequency  # fb num = fb_frq*trial_length