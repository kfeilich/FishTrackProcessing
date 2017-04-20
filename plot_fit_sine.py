import peakutils
import scipy
from scipy.optimize import leastsq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_fit_sine(tracklist_subset, tracklist, sin_corr_df):
    """Plots detrended tail tip motion against best fit sine wave

       Does what it says above, using the data from tracklist and 
       sin_corr_df, including an annotation with the Pearson's 
       correlation coefficient between the two traces.

               Args:
                   tracklist_subset (list): a subset of trials, typically 
                       generated using the convenience function 
                       make_subset()
                   tracklist (dict): a tracklist produced by extract_data()
                   sin_corr_df (dataframe): a Pandas dataframe 
                        produced by sin_corr() containing the trials in 
                        tracklist subset
                        
                Returns: None, just a plot

              """

    for trial in tracklist_subset:
        raw_data = tracklist[trial]['data']['pt2y']
        behavior = tracklist[trial]['behavior']
        init_speed = tracklist[trial]['start_spd_BLs']
        base = peakutils.baseline(raw_data, 3)  # Find bkgrd trend
        raw_data = raw_data-base
        time = raw_data.index.values

        amp = sin_corr_df['Est.Amplitude'][trial]
        freq = sin_corr_df['Est.Freq'][trial]
        phase = sin_corr_df['Est.Phase'][trial]
        offset = sin_corr_df['Est.Offset'][trial]
        cor_coeff = str(np.round(sin_corr_df['Pearsons'][trial], 2))
        pvalue = str(np.round(sin_corr_df['Pvalue'][trial], 3))
        annotation = str('r = ' + cor_coeff + '\np = ' + pvalue)

        data_fit = amp * np.sin(2 * np.pi * freq * time + phase) + offset

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle(trial + ' ' + behavior + ' ' + str(init_speed) + 'BL/s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (cm)')
        ax.plot(time, raw_data, label = 'Raw Data')
        ax.plot(time, data_fit, label="Sine Wave Fit")
        ax.text(0.02, 0.98, annotation, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)
        ax.legend()
        plt.show()
