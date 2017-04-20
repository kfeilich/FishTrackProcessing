import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import peakutils

matplotlib.rc('axes.formatter', useoffset=False)


def plot_fft_forfig(rows, columns, number,
                    trial, tracklist):
    """Plots fast Fourier transfrom of detrended tailtip motion
    
    Does what it says above. Used to make panels for a composite 
    figure (see Figure 5 for example.)

        Args:
            rows (int): Number of rows in composite figure
            columns (int): Number columns in composite figure
            number (int): Number of this subplot
            trial (str): a strings with the trial name
            tracklist (dict): a tracklist produced by extract_data()
            
        Returns:
            ax (matplotlib Axes)
        """

    trial_name = tracklist[trial]['sequence']
    # print(trial_name)  # for diagnostics
    fish = tracklist[trial]['fish']
    tailtip = tracklist[trial]['data']['pt2y_smth']
    time = tracklist[trial]['data'].index.values
    base = peakutils.baseline(tailtip, 1)  # Find linear bkgrd trend
    framerate = tracklist[trial]['FPS']
    behavior = tracklist[trial]['behavior']

    if fish == 'Bass1':
        col = 'cornflowerblue'
    else:
        col = 'salmon'

    # Do the FFT using Scipy
    tailbeat_freqs = np.fft.fft(tailtip - base)
    frequency_domain = np.fft.fftfreq(len(time), 1 / framerate)

    ax = fig3.add_subplot(rows, columns, number)
    ax.plot(np.abs(frequency_domain),
            np.abs(tailbeat_freqs), col)
    ax.set_xlim(0, 12)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')

    return ax
