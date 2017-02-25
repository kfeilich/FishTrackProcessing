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
