import peakutils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_fit_sine_forfig(nrows, ncols, number, trial, tracklist,
                         sin_corr_df):
    """Plots detrended tail tip motion against best fit sine wave

       Does what it says above, using the data from tracklist and 
       sin_corr_df, including an annotation with the Pearson's 
       correlation coefficient between the two traces. 
       
       Note: This figure is used to make subpanels for composite 
       figures. See Figure 4 from the paper.
       

               Args:
                   nrows (int): Number of rows in composite figure
                   ncols (int): Number columns in composite figure
                   number (int): Number of this subplot
                   tracklist_subset (list): a subset of trials, typically 
                       generated using the convenience function 
                       make_subset()
                   tracklist (dict): a tracklist produced by extract_data()
                   sin_corr_df (dataframe): a Pandas dataframe 
                        produced by sin_corr() containing the trials in 
                        tracklist subset

                Returns: None, just a plot

              """

    raw_data = tracklist[trial]['data']['pt2y']
    fish = tracklist[trial]['fish']
    base = peakutils.baseline(raw_data, 3)  # Find bkgrd trend
    raw_data = raw_data - base
    time = raw_data.index.values

    if fish == 'Bass1':
        col1 = 'darkblue'
        col2 = 'cornflowerblue'
    else:
        col1 = 'darkred'
        col2 = 'salmon'

    amp = sin_corr_df['Est.Amplitude'][trial]
    freq = sin_corr_df['Est.Freq'][trial]
    phase = sin_corr_df['Est.Phase'][trial]
    offset = sin_corr_df['Est.Offset'][trial]
    cor_coeff = str(np.round(sin_corr_df['Pearsons'][trial], 2))
    pvalue = str(np.round(sin_corr_df['Pvalue'][trial], 3))
    annotation = str('r = ' + cor_coeff + '\np = ' + pvalue)

    data_fit = amp * np.sin(2 * np.pi * freq * time + phase) + offset

    ax = fig4.add_subplot(nrows, ncols, number)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (cm)')
    ax.plot(time, raw_data, col1, label='Tail Tip Data', lw=3)
    ax.plot(time, data_fit, col2, label="Sine Wave Fit", lw=2)
    t = ax.text(0.02, 0.97, annotation, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)
    t.set_bbox(dict(alpha=0.8, edgecolor='black', facecolor='white'))

    return ax