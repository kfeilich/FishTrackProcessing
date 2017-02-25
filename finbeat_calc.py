from __future__ import division, print_function
# Load all of the things
import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import peakutils
from peakutils.plot import plot as pplot

matplotlib.rc('axes.formatter', useoffset=False)

root = tk.Tk()
root.withdraw()

def finbeat_calc(tracklist_subset, tracklist):
    """Determines finbeat peaks, troughs, and timings for each trial

    This function takes the trial information and data for a
    subset of the trials in a tracklist compiled by extract_data(),
    and finds the fin beat peaks and troughs in the tail (pt 2)
    position data. Using the module peakutils, this function
    prepares the position data for peak finding by fitting and
    removing a linear baseline for each trial, and produces indices
    and values for the peaks and troughs. It returns a dictionary
    containing the timing, magnitude, and type of the local extrema
    for each trial.

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
        finbeats[trial_name] (dict): each key referring to its own
        dictionary with the following entries:
            finbeats[trial_name]['trial_name'](str): name of trial
            finbeats[trial_name]['behavior'](str): behavior as coded by
                                                filmer.
                                                'A': linear acc.
                                                'B': burst acc.
                                                'S': steady swim
            finbeats[trial_name]['fish'] (str): name of fish
            finbeats[trial_name]['fb_data'] (DataFrame): fin beat data
                                            with the following columns:
                    ['fb_data']['time'] (floats): timestamp in seconds
                    ['fb_data']['ypos'] (floats): pos. of tail tip in cm
                    ['fb_data']['type'] (str): denotes phase of finbeat.
                                                'P' = peak
                                                'T' = trough
    """

    finbeats = {}

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

        # First shot with peakutils
        clean_tailtip = tailtip.reset_index(drop=True)
        trough_tailtip = -clean_tailtip
        peak_indexes = peakutils.indexes(clean_tailtip - base,
                                         thres=0.3, min_dist=50)
        trough_indexes = peakutils.indexes(trough_tailtip + base,
                                           thres=0.3, min_dist=50)

        # print(peak_indexes)
        # print(time[peak_indexes], tailtip[time[peak_indexes]])
        # print(trough_indexes)
        # print(time[trough_indexes], tailtip[time[trough_indexes]])
        # plt.figure(figsize=(10, 6))
        # pplot(time, clean_tailtip - base, peak_indexes)
        # pplot(time, clean_tailtip - base, trough_indexes)
        # plt.show()

        # Organize the output data
        peaks = {'time': time[peak_indexes],
                 'ypos': clean_tailtip[peak_indexes],
                 'type': ['P']*len(time[peak_indexes])}
        peak_df = pd.DataFrame(peaks)
        troughs = {'time': time[trough_indexes],
                   'ypos': clean_tailtip[trough_indexes],
                   'type': ['T'] * len(time[trough_indexes])}
        trough_df = pd.DataFrame(troughs)

        fbs = pd.concat([peak_df,trough_df], ignore_index=True)
        fbs = fbs.sort_values(by='time').reset_index(drop=True)

        finbeats[trial_name] ={'trial_name': trial_name,
                               'behavior': behavior, 'fish': fish,
                               'fb_data': fbs}
    return finbeats
