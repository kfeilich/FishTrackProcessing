import numpy as np
import pandas as pd
import scipy
import math
import peakutils

def cross_corr(subset, tracklist):
    """Calculates max. cross-correlation btw pairs of tail tip FFTs

        This function calculates the fast fourier transforms (FFT) of 
        detrended tail tip data for a list of trials, and then 
        calculates the maximum cross-correlation between each pair of 
        FFTs. 

        Args:
            subset (list): a subset of trials, typically generated 
            using the convenience function make_subset()
            tracklist (dict): a tracklist produced by extract_data()

        Returns:
            cross_corr_mat (Dataframe): column and row indices are 
            identical (the trial names), entries are the max. 
            cross-correlation between the pair of trials. 


        """
    # Initialize the output dataframe
    cross_corr_mat = pd.DataFrame(data=None, index=subset,
                              columns=subset)
    # Iterate over the list of trials to calculate FFT
    for i in subset:
        tailtip = tracklist[i]['data']['pt2y_smth']
        base = peakutils.baseline(tailtip, 3)  # Find linear bkgrd trend
        tailtip = tailtip-base

    # Iterate over the list of trials to establish pairs
        for j in subset:
            tailtip2 = tracklist[j]['data']['pt2y_smth']
            base2 = peakutils.baseline(tailtip2, 3)
            tailtip2 = tailtip2-base2

            # Pad the data appropriately
            if len(tailtip) < len(tailtip2):
                pad_tot = len(tailtip2) - len(tailtip)
                if pad_tot % 2 == 0:
                    pad = int(pad_tot / 2.0)
                    tailtip = np.pad(tailtip, (pad, pad), 'mean')
                else:
                    pad = int(math.ceil(pad_tot / 2.0))
                    tailtip= np.pad(tailtip, (pad, pad - 1),
                                          'mean')

            elif len(tailtip2) < len(tailtip):
                pad_tot = len(tailtip) - len(tailtip2)
                if pad_tot % 2 == 0:
                    pad = int(pad_tot / 2.0)
                    tailtip2 = np.pad(tailtip2, (pad, pad), 'mean')
                else:
                    pad = int(math.ceil(pad_tot / 2.0))
                    tailtip2 = np.pad(tailtip2, (pad, pad - 1), 'mean')

            # Do the FFT using Scipy
            tailbeat_FFT = np.abs(np.fft.fft(tailtip))
            tailbeat_FFT2 = np.abs(np.fft.fft(tailtip2))

            tailbeat_FFT = np.delete(tailbeat_FFT, np.arange(0,100), 0)
            tailbeat_FFT2 = np.delete(tailbeat_FFT2, np.arange(0,100),0)

            # Standardize for cross-correlation
            tailbeat_FFT_norm = (tailbeat_FFT - np.mean(tailbeat_FFT)) / (
            np.std(tailbeat_FFT) * len(tailbeat_FFT))
            tailbeat_FFT2_norm = (tailbeat_FFT2 - np.mean(
                tailbeat_FFT2)) / np.std(tailbeat_FFT2)

            # Calculate cross-correlation using SciPy
            corr_max = scipy.signal.correlate(tailbeat_FFT_norm,
                                              tailbeat_FFT2_norm,
                                              mode='valid')
            # Get the maximum value of the cross-correlation.
            max_index = np.argmax(corr_max)
            cross_corr_mat[i][j] = corr_max[max_index]

    cross_corr_mat.dropna(axis=(0, 1), how="all", inplace=True)
    cross_corr_mat = cross_corr_mat.apply(pd.to_numeric)

    return cross_corr_mat