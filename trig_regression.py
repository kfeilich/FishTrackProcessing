import numpy as np
from scipy.optimize import curve_fit


def sinusoidal_regression(X, y, bounds=None):
    """ Model the relationship between X and y as a sinusoidal relationship.  This is perfomed via an optimization
    routine which allows for a resulting covariance matrix. However, as optimization can be sensitive to initialization,
    a basic set of rules is used to get the starting coefficients.

    * amplitude = max(data) - min(data)
    * offset    = max(data) - (amplitude)/2.0
    * phase     = 0 #TODO, come up with a good way to do this
    :param X:
    :param y
    :return: curve_fit results.  popt, pcov
    """

    def objective(my_X, a,b, frequency, offset):
        return a * np.sin(frequency*my_X) +b*np.cos(frequency*my_X) + offset

    def jacobian(my_X, a,b, frequency, offset):
        cos_freq_x = np.cos(frequency*my_X)
        sin_freq_x = np.sin(frequency*my_X)
        sub_a = sin_freq_x
        sub_b = cos_freq_x
        #cos_sin_x = np.cos(my_X)*np.sin(my_X)
        sub_freq = a*my_X*cos_freq_x - b * my_X *sin_freq_x#(a-b)*(frequency* cos_sin_x)
        sub_offset = np.ones_like(my_X)

        jac = [sub_a, sub_b, sub_freq, sub_offset]
        jac = [np.reshape(sub,(-1,1)) for sub in jac]
        return np.concatenate(jac,axis=1)

    max_y = np.max(y)
    min_y = np.min(y)
    amplitude0 = max_y - min_y
    offset0 = max_y - amplitude0/2.0
    frequency0= 0.01
    #phase0 = 0.0
    coefs, covar = curve_fit(objective, X, y, p0=np.array([-amplitude0,amplitude0, frequency0, offset0]), jac=jacobian,bounds=bounds, max_nfev=int(1e5))
    return coefs, covar

def windowed_sinusoid(X, y, window_size=101):
    """ Calculate rolling sinusoid regression parameters.

    :param X:
    :param y:
    :param window_size:
    :return: coeff_array, cov_array of length len(y)-window_size+1
    """

    coefficients = []
    covariances = []

    max_length = len(y) - window_size + 1
    bounds = [[- np.inf, -np.inf, -np.inf, np.min(y)], [np.inf, np.inf, np.inf, np.max(y)]]
    for offset in range(max_length):
        #print(offset)
        coefficients_o, covariances_o = sinusoidal_regression(X[offset:offset+window_size], y[offset:offset+window_size], bounds=bounds)
        coefficients.append(coefficients_o)
        covariances.append(covariances_o)

    return np.array(coefficients), np.array(covariances)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import pandas

    data = pandas.read_csv('Data/Bass1_S13_tracks_xypts.csv')


    y= data['pt2_cam2_Y'].values
    X = np.arange(len(y))
    window_size = 701
    fig, ax = plt.subplots(2,2)
    coefficients, covariances = windowed_sinusoid(X, y, window_size)
    ax[0][0].plot(X, y, color='m')

    padding = int(np.floor(window_size/2))
    pad_zero = np.zeros((padding,coefficients.shape[1]))
    padded = np.concatenate([pad_zero, coefficients, pad_zero])
    amplitudes = np.sqrt(padded[:,0]**2+padded[:,1]**2)
    frequency = padded[:,2]
    phase = np.arctan2(padded[:,1], padded[:,0])
    offset = padded[:, 3]

    ax[0][0].plot(X, amplitudes, color='g')
    ax[0][0].plot(X, phase, color='r')
    ax[0][0].plot(X, offset, color='b')
    ax[0][0].plot(X,amplitudes*np.sin(X*frequency+phase)+offset, color='k')

    amplitudes_un = np.sqrt(coefficients[:,0]**2+coefficients[:,1]**2)

    ax[0][1].hist(amplitudes_un)
    ax[1][0].hist(coefficients[:,2])
    fig.show()