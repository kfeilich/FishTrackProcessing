import peakutils
import scipy
from scipy.optimize import leastsq
import pandas as pd
import numpy as np
from scipy import signal


def sin_corr(tracklist_subset, tracklist, finbeat_df):
    """Computes full width at half maximum from FFT of tailbeat movement

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
            finbeat_df (dict): a finbeat dataframe produced by finbeat_calc
        Returns:


"""
    # Initialize Dataframe to Contain Trial Sine Wave Estimates and Pearsons R
    sincorr_df = pd.DataFrame(columns=['Behavior', 'InitSpd',
                                       'Pearsons',
                                       'Pvalue',
                                       'Est.Freq', 'Est.Amplitude',
                                       'Est.Phase', 'Est.Offset'],
                              index=tracklist_subset)

    # Iterate over all of the trials
    for trial in tracklist_subset:
        # Pull data for each trial
        trialname = tracklist[trial]['sequence']
        behavior = tracklist[trial]['behavior']
        init_spd = tracklist[trial]['start_spd']
        data = tracklist[trial]['data']['pt2y']
        base = peakutils.baseline(data, 3)  # Find bkgrd trend
        data = data - base  # remove background trend
        time = tracklist[trial]['data'].index.values
        periods = np.array(finbeat_df[trial]['period'])

        # Get first estimates for optimization from data
        # Estimate frequency
        periods = np.sort(periods)
        periods = periods[np.logical_not(np.isnan(periods))]
        periods = np.delete(periods, [0, len(periods) - 1])
        period = np.mean(periods)
        guess_frequency = 1. / period
        # Estimate amplitude ( = 1/2 excursion)
        amplitudes = np.array(finbeat_df[trial]['amplitude'])
        amplitudes = abs(amplitudes)
        amplitudes = np.sort(amplitudes)
        amplitudes = amplitudes[np.logical_not(np.isnan(amplitudes))]
        amplitudes = np.delete(amplitudes, [0, len(amplitudes) - 1])
        amplitude = np.mean(amplitudes) / 2
        guess_amplitude = amplitude
        #Estimate y-axis and temporal shifts (arbitrary)
        guess_offset = np.mean(data)
        guess_phase = 0

        # Compose first guess sine wave (remembering sine wave freq = angular)
        data_first_guess = guess_amplitude * np.sin(
            2 * np.pi * guess_frequency * time+ guess_phase) + guess_offset

        # Optimize the estimates by minimizing the least squares distance btw
        # estimate and data. Return estimates of best fit amplitude, frequency,
        # phase, and offset
        optimize_func = lambda x: x[0] * np.sin(
            2 * np.pi * x[1] * time + x[2]) + x[3] - data
        est_amplitude, est_frequency, est_phase, est_offset = leastsq(
            optimize_func, [guess_amplitude, guess_frequency,
                            guess_phase, guess_offset])[0]

        # Create best-fit sin wave from estimated parameters
        data_fit = est_amplitude * np.sin(
            2 * np.pi * est_frequency * time + est_phase) + est_offset
        # Correlate the actual trial data with the simulated best-fit sine wave
        corr = scipy.stats.pearsonr(data, data_fit)

        # Add the estimates and correlation statistics to the output dataframe
        sincorr_df.loc[trialname] = pd.Series(
            {'Behavior': behavior, 'InitSpd': init_spd,
             'Pearsons': corr[0], 'Pvalue': corr[1],
             'Est.Freq': est_frequency, 'Est.Amplitude': est_amplitude,
             'Est.Phase': est_phase, 'Est.Offset': est_offset})
    return sincorr_df
