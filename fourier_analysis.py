import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import peakutils

matplotlib.rc('axes.formatter', useoffset=False)


def fourier_analysis(tracklist_subset, tracklist):
    """Computes FFT of tailbeat movement (do not use)

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
            None, just a plot.


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

        # Do the FFT using Scipy
        tailbeat_freqs = np.fft.fft(tailtip - base)
        frequency_domain = np.fft.fftfreq(len(time), 1 / framerate)

        fig = plt.figure()
        plt.suptitle(trial)
        plt.plot(np.abs(frequency_domain),
                 np.abs(tailbeat_freqs))
        plt.xlim(0,12)
        plt.figure()

