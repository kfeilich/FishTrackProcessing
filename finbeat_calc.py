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
        finbeat_byP[trial_name] (dict): each key referring to its own
        dictionary with the following entries for fb defined by PEAKS
        only, in addition to those pulled from the finbeats dict:
            finbeat_byP[trial_name]['endtime'] (float): end of cycle
            finbeat_byP[trial_name]['period'] (float): length of finbeat
            finbeat_byP[trial_name]['amplitude'] (float): peak-to-trough
        finbeat_byT[trial_name] (dict): same as for finbeat_byP,
        except entries for fb defined by TROUGHS only
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

    # Calculate parameters for each finbeat from peaks and from troughs
    finbeat_byP = {}
    finbeat_byT = {}

    # Iterate for each trial
    for trial in tracklist_subset:
        # Pull trial finbeat data
        trial_name = tracklist[trial]['sequence']
        fb_data = finbeats[trial]['fb_data']

        # Initialize new dataframes for the finbeat-specific measures
        fb_peak_params = pd.DataFrame()
        fb_trough_params = pd.DataFrame()

        # Calculate parameters w/ peaks as starting point
        peaksmask = fb_data.loc[:, 'type'] == 'P'
        fb_peak_params = fb_data.loc[peaksmask]
        #TODO: Replace these with appropriate .loc operators
        fb_peak_params['endtime'] = np.nan
        fb_peak_params['period'] = np.nan
        fb_peak_params['nxttrough'] = np.nan
        fb_peak_params['amplitude'] = np.nan

        # Iterate for each peak in trial
        for i in fb_peak_params.index.values:
            # If there is a subsequent peak, use it for endtime, period
            if i + 2 in fb_peak_params.index.values:
                fb_peak_params.loc[i, 'endtime'] = fb_peak_params.loc[
                    i + 2, 'time']
                fb_peak_params.loc[i, 'period'] = fb_peak_params.loc[
                                                      i, 'endtime'] - \
                                                  fb_peak_params.loc[
                                                      i, 'time']
            else:
                fb_peak_params.loc[i, 'endtime'] = np.nan
                fb_peak_params.loc[i, 'period'] = np.nan

            # If there is a subsequent trough, use it to get amplitude
            if i + 1 in fb_data.index.values:
                fb_peak_params.loc[i, 'nxttrough'] = fb_data.loc[
                    i + 1, 'ypos']
                fb_peak_params.loc[i, 'amplitude'] = abs(
                    fb_peak_params.loc[i, 'nxttrough'] -
                    fb_peak_params.loc[i, 'ypos'])
            else:
                fb_peak_params.loc[i, 'nxttrough'] = np.nan
                fb_peak_params.loc[i, 'amplitude'] = np.nan

        # Store the results of the iterations
        finbeat_byP[trial_name] = fb_peak_params

        # Iterate for each trough in trial
        troughsmask = fb_data.loc[:, 'type'] == 'T'
        fb_trough_params = fb_data.loc[troughsmask]
        fb_trough_params['endtime'] = np.nan
        fb_trough_params['period'] = np.nan
        fb_trough_params['nxtpeak'] = np.nan
        fb_trough_params['amplitude'] = np.nan

        # If there is a subsequent trough, use it for endtime, period
        for i in fb_trough_params.index.values:
            if i + 2 in fb_trough_params.index.values:
                fb_trough_params.loc[i, 'endtime'] = \
                fb_trough_params.loc[i + 2, 'time']
                fb_trough_params.loc[i, 'period'] = \
                fb_trough_params.loc[i, 'endtime'] - \
                fb_trough_params.loc[i, 'time']
            else:
                fb_trough_params.loc[i, 'endtime'] = np.nan
                fb_trough_params.loc[i, 'period'] = np.nan

            # If there is a subsequent peak, use it to get amplitude
            if i + 1 in fb_data.index.values:
                fb_trough_params.loc[i, 'nxtpeak'] = fb_data.loc[
                    i + 1, 'ypos']
                fb_trough_params.loc[i, 'amplitude'] = \
                fb_trough_params.loc[i, 'nxtpeak'] - \
                fb_trough_params.loc[i, 'ypos']
            else:
                fb_trough_params.loc[i, 'nxtpeak'] = np.nan
                fb_trough_params.loc[i, 'amplitude'] = np.nan

        # Store the results of the iterations
        finbeat_byT[trial_name] = fb_trough_params

    return finbeats, finbeat_byP, finbeat_byT
