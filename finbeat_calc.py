from __future__ import division, print_function
# Load all of the things
import tkinter as tk
import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.cm as cm
import matplotlib
import sys
import matplotlib.pyplot as plt
import peakutils
from scipy.signal import savgol_filter  # for smoothing data
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog  # For folder input popup
from peakutils.plot import plot as pplot
matplotlib.rc('axes.formatter', useoffset=False)

root = tk.Tk()
root.withdraw()

def finbeat_calc(tracklist_subset, tracklist):
    tail_amplitudes = {}
    tail_periods = {}

    for trial in tracklist_subset:  # Iterate over desired trials
        trial_name = tracklist[trial]['sequence']
        print(trial_name)
        fish = tracklist[trial]['fish']
        tailtip = tracklist[trial]['data']['pt2y_smth']
        time = tracklist[trial]['data'].index.values
        base = peakutils.baseline(tailtip, 1)  # Find linear bkgrd trend
        framerate = tracklist[trial]['FPS']

        # Find best guess for number of peaks using FFT
        fourier = np.fft.fft(tailtip)
        frequencies = np.fft.fftfreq(len(time), 1/framerate)
        positive_frequencies = frequencies[np.where(frequencies >= 0)]
        magnitudes = abs(fourier[np.where(frequencies >= 0)])
        peak_frequency = np.argmax(magnitudes)
        guess = max(time)*peak_frequency  # fb num = fb_frq*trial_length

        # First shot with peakutils
        clean_tailtip = tailtip.reset_index(drop = True)
        indexes = peakutils.indexes(clean_tailtip-base, thres=0.3, min_dist=50)
        print(indexes)
        print(time[indexes], tailtip[time[indexes]])
        plt.figure(figsize=(10, 6))
        pplot(time, clean_tailtip-base, indexes)