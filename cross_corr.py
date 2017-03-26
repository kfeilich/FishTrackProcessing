import numpy as np
import pandas as pd
import scipy
import math
import peakutils

def cross_corr(subset, tracklist):

    cross_corr_mat = pd.DataFrame(data=None, index=subset,
                              columns=subset)

    for i in subset:
        tailtip = tracklist[i]['data']['pt2y_smth']
        base = peakutils.baseline(tailtip, 3)  # Find linear bkgrd trend

        # Do the FFT using Scipy
        tailbeat_FFT = np.abs(np.fft.fft(tailtip - base))

        for j in subset:
            tailtip2 = tracklist[j]['data']['pt2y_smth']
            base2 = peakutils.baseline(tailtip2,
                                       3)  # Find linear bkgrd trend

            # Do the FFT using Scipy
            tailbeat_FFT2 = np.abs(np.fft.fft(tailtip2 - base2))

            # Pad the FFTs
            if len(tailbeat_FFT) < len(tailbeat_FFT2):
                pad_tot = len(tailbeat_FFT2) - len(tailbeat_FFT)
                if pad_tot % 2 == 0:
                    pad = int(pad_tot / 2.0)
                    tailbeat_FFT = np.pad(tailbeat_FFT, (pad, pad), 'mean')
                else:
                    pad = int(math.ceil(pad_tot / 2.0))
                    tailbeat_FFT = np.pad(tailbeat_FFT, (pad, pad - 1),
                                          'mean')

            elif len(tailbeat_FFT2) < len(tailbeat_FFT):
                pad_tot = len(tailbeat_FFT) - len(tailbeat_FFT2)
                if pad_tot % 2 == 0:
                    pad = int(pad_tot / 2.0)
                    tailbeat_FFT2 = np.pad(tailbeat_FFT2, (pad, pad),
                                           'mean')
                else:
                    pad = int(math.ceil(pad_tot / 2.0))
                    np.pad(tailbeat_FFT, (pad, pad - 1), 'mean')

            tailbeat_FFT_norm = (tailbeat_FFT - np.mean(tailbeat_FFT)) / (
            np.std(tailbeat_FFT) * len(tailbeat_FFT))
            tailbeat_FFT2_norm = (tailbeat_FFT2 - np.mean(
                tailbeat_FFT2)) / np.std(tailbeat_FFT2)

            corr_max = scipy.signal.correlate(tailbeat_FFT_norm,
                                              tailbeat_FFT2_norm,
                                              mode='valid')
            max_index = np.argmax(corr_max)
            # print(i, j, max_index, np.around(corr_max[max_index], decimals = 3))
            # cross_corr[i][j] = (np.around(corr_max[max_index], decimals = 3), max_index)
            cross_corr_mat[i][j] = corr_max[max_index]

    cross_corr_mat.dropna(axis=(0, 1), how="all", inplace=True)
    cross_corr_mat = cross_corr_mat.apply(pd.to_numeric)

    return cross_corr_mat