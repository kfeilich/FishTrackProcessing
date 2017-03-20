import peakutils
from scipy.optimize import leastsq

def sin_corr(tracklist_subset, tracklist):
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
        Returns:


"""
    for trial in tracklist_subset:
        data = tracklist[trial]['data']['pt2y_smth']
        framerate = tracklist[trial]['FPS']
        base = peakutils.baseline(data, 3)  # Find linear bkgrd trend
        data = data - base
        time = tracklist[trial]['data'].index.values
        t = time
        periods = np.array(finbeat_byP[trial]['period'])
        # periods = np.append(amplitudes,
        #                       np.array(finbeat_byT[trial]['period']))
        periods = np.sort(periods)
        periods = periods[np.logical_not(np.isnan(periods))]
        periods = np.delete(periods, [0, len(periods) - 1])
        period = np.mean(periods)
        print(periods)
        peak_frequency = 1. / period

        amplitudes = np.array(finbeat_byP[trial]['amplitude'])
        amplitudes = np.append(amplitudes,
                               np.array(finbeat_byT[trial]['amplitude']))
        amplitudes = abs(amplitudes)
        amplitudes = np.sort(amplitudes)
        amplitudes = amplitudes[np.logical_not(np.isnan(amplitudes))]
        amplitudes = np.delete(amplitudes, [0, len(amplitudes) - 1])
        amplitude = np.mean(amplitudes) / 2
        guess_offset = np.mean(data)
        guess_phase = 0

        data_first_guess = amplitude * np.sin(
            2 * np.pi * peak_frequency * t) + guess_offset

        optimize_func = lambda x: amplitude * np.sin(
            2 * np.pi * peak_frequency * t + x[0]) + guess_offset - data
        est_phase = leastsq(optimize_func, [guess_phase])[0]

        data_fit = amplitude * np.sin(
            2 * np.pi * peak_frequency * t + est_phase) + guess_offset
        print(est_phase)
        print(amplitude)
        print(peak_frequency)

        plt.plot(time, data, '.')
        plt.plot(time, data_fit, label='after fitting')
        plt.legend()
        plt.show()
