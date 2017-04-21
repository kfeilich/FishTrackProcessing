from __future__ import division, print_function
import tkinter as tk
import os
import sys
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import math
import peakutils
from peakutils.plot import plot as pplot
import pickle
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.rc('axes.formatter', useoffset=False)

def boot_outputs(model, boot_list, interact=True, amplitude=True):
    """Takes bootstrap replicates and returns 95% CIs and pseudo-p-vals 

           This function takes the set of bootstrap replicates 
           produced by mult_reg_boot, and returns 95% CIs for the 
           regression coefficients and pseudo-p-values for the 
           initial base model. 

           Args:
               model (statsmodels RegressionResults): the initial 
                    model output using all data points
               boot_list (list): a list of RegressionResults 
                    produced by mult_reg_boot().
               interact (Bool): True if the initial model have an 
                    interaction effect, False if not. 
               amplitude (Bool): True if the initial model 
                    included a coefficient for amplitude, False if not. 

           Returns:
               boot_output (DataFrame): Columns are 'Pseudo Pvalues' 
                    and '95% CIs', rows are the parameters being 
                    estimated.


           """
    # Setting up the output dataframe
    model_params = ['F_test', 'Intercept', 'Period']
    if amplitude == True:
        model_params.append('Amplitude')
    if interact == True:
        model_params.append('Interact')
    boot_output = pd.DataFrame(data=None, columns=['pvalues', 'CIs95'],
                               index=model_params)

    # Return original parameter estimates
    coef_obs = model.params

    #  Find bootstrap F pvalue for whole test
    fobs = model.fvalue
    fvals = []
    for i in boot_list:
        fvals.append(i.fvalue)
    fvals_high = sum(j > fobs for j in fvals)
    boot_output['pvalues']['F_test'] = fvals_high / len(fvals)

    # Find Bootstrap p-vals for Coefficients
    # Pull all bootstrap estimates
    pvals = []
    for i in boot_list:
        pvals.append(i.pvalues)
    coefs = []
    for i in boot_list:
        coefs.append(i.params)
    tvals = []
    for i in boot_list:
        tvals.append(i.tvalues)

    # Initialize outputs
    if interact == True:
        Interact_coefs = []
        Interact_pvals = []
        Interact_tvals = []
    Period_coefs = []
    Period_pvals = []
    Period_tvals = []
    if amplitude == True:
        Amplitude_coefs = []
        Amplitude_pvals = []
        Amplitude_tvals = []
    Intercept_coefs = []
    Intercept_pvals = []
    Intercept_tvals = []

    for i in pvals:
        if interact == True:
            Interact_pvals.append(i['Period:Amplitude'])
        Period_pvals.append(i['Period'])
        if amplitude == True:
            Amplitude_pvals.append(i['Amplitude'])
        Intercept_pvals.append(i['Intercept'])

    for i in coefs:
        if interact == True:
            Interact_coefs.append(i['Period:Amplitude'])
        Period_coefs.append(i['Period'])
        if amplitude == True:
            Amplitude_coefs.append(i['Amplitude'])
        Intercept_coefs.append(i['Intercept'])

    # Confidence Intervals
    Intercept_coefs = sorted(Intercept_coefs)
    d_Intercept_coefs = Intercept_coefs - coef_obs['Intercept']
    quant_Intercept = np.percentile(a=d_Intercept_coefs, q=[2.5, 97.5])
    ci_Intercept = np.subtract(coef_obs['Intercept'], quant_Intercept)
    boot_output['CIs95']['Intercept'] = ci_Intercept

    Period_coefs = sorted(Period_coefs)
    d_Period_coefs = Period_coefs - coef_obs['Period']
    quant_Period = np.percentile(a=d_Period_coefs, q=[2.5, 97.5])
    ci_Period = np.subtract(coef_obs['Period'], quant_Period)
    boot_output['CIs95']['Period'] = ci_Period

    if amplitude == True:
        Amplitude_coefs = sorted(Amplitude_coefs)
        d_Amplitude_coefs = Amplitude_coefs - coef_obs['Amplitude']
        quant_Amplitude = np.percentile(a=d_Amplitude_coefs,q=[2.5,
                                                               97.5])
        ci_Amplitude = np.subtract(coef_obs['Amplitude'],
                                   quant_Amplitude)
        boot_output['CIs95']['Amplitude'] = ci_Amplitude

    if interact == True:
        Interact_coefs = sorted(Interact_coefs)
        d_Interact_coefs = Interact_coefs - coef_obs['Period:Amplitude']
        quant_Interact = np.percentile(a=d_Interact_coefs, q=[2.5,
                                                              97.5])
        ci_Interact = np.subtract(coef_obs['Period:Amplitude'],
                                  quant_Interact)
        boot_output['CIs95']['Interact'] = ci_Interact

    # P- Values
    p1_Intercept = sum(j > 0.0 for j in Intercept_coefs) / len(
        Intercept_coefs)
    p2_Intercept = sum(j < 0.0 for j in Intercept_coefs) / len(
        Intercept_coefs)
    boot_output['pvalues']['Intercept'] = min(p1_Intercept,
                                              p2_Intercept) * 2

    p1_Period = sum(j > 0.0 for j in Period_coefs) / len(
        Period_coefs)
    p2_Period = sum(j < 0.0 for j in Period_coefs) / len(
        Period_coefs)
    boot_output['pvalues']['Period'] = min(p1_Period, p2_Period) * 2

    if amplitude == True:
        p1_Amplitude = sum(j > 0.0 for j in Amplitude_coefs) / len(
            Amplitude_coefs)
        p2_Amplitude = sum(j < 0.0 for j in Amplitude_coefs) / len(
            Amplitude_coefs)
        boot_output['pvalues']['Amplitude'] = min(p1_Amplitude,
                                                  p2_Amplitude) * 2

    if interact == True:
        p1_Interact = sum(j > 0.0 for j in Interact_coefs) / len(
            Interact_coefs)
        p2_Interact = sum(j < 0.0 for j in Interact_coefs) / len(
            Interact_coefs)
        boot_output['pvalues']['Interact'] = min(p1_Interact,
                                                 p2_Interact) * 2

    return boot_output

def check_plots(tracklist_subset, tracklist):
    """Plot some diagnostics from trial data produced by extract_data()

    This is just a convenience plotting function to have a look at 
    the position, velocity, and acceleration data after initial 
    processing using extract_data. It is useful for sanity checks, 
    making sure everything looks reasonable. 

    Args:
       tracklist_subset (list): 
           List of strings indicating sequence names of desired trials.
       tracklist (dict):
           tracklist produced by extract_data()

    Returns
        None. Just plots.
          """

    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)
        data = tracklist[trial]['data']
        colors = plt.cm.cubehelix(scaled_time)
        m = cm.ScalarMappable(cmap=cm.cubehelix)
        m.set_array(tracklist[trial]['data'].index.values)

        # filename = str(trial) + '.pdf'

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist[trial]['behavior'])

        ax1 = fig.add_subplot(4, 2, 1, projection='3d')
        ax1.set_title('Pt 1 Position')
        ax1.scatter3D(xs=tracklist[trial]['data']['pt1x_smth'],
                      ys=tracklist[trial]['data']['pt1y_smth'],
                      zs=tracklist[trial]['data']['pt1z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax1.autoscale(enable=True, tight=True)
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        ax2 = fig.add_subplot(4, 2, 3)
        ax2.plot(data.index.values, data['pt1y_v_smth'], 'bo')
        ax2.set_ylabel('Y velocity (cm/s)', color='b')
        ax2.tick_params('y', colors='b')

        ax3 = fig.add_subplot(4, 2, 5)
        ax3.plot(data.index.values, data['pt1_net_v'], 'bo')
        ax3.set_ylabel('Net Velocity (cm/s)', color='b')
        ax3.tick_params('y', colors='b')

        ax4 = fig.add_subplot(4, 2, 7)
        ax4.plot(data.index.values, data['pt1_net_a'], 'bo')
        ax4.set_ylabel('Net accel (cm/s2)', color='b')
        ax4.tick_params('y', colors='b')

        ax5 = fig.add_subplot(4, 2, 2, projection='3d')
        ax5.set_title('Pt 2 Position')
        ax5.scatter3D(xs=tracklist[trial]['data']['pt2x_smth'],
                      ys=tracklist[trial]['data']['pt2y_smth'],
                      zs=tracklist[trial]['data']['pt2z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax5.autoscale(enable=True, tight=True)
        ax5.set_xlabel('X position')
        ax5.set_ylabel('Y position')
        ax5.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        ax6 = fig.add_subplot(4, 2, 4)
        ax6.plot(data.index.values, data['pt2y_v_smth'], 'bo')
        ax6.set_ylabel('Y velocity (cm/s)', color='b')
        ax6.tick_params('y', colors='b')

        ax7 = fig.add_subplot(4, 2, 6)
        ax7.plot(data.index.values, data['pt2_net_v'], 'bo')
        ax7.set_ylabel('Net Velocity (cm/s)', color='b')
        ax7.tick_params('y', colors='b')

        ax8 = fig.add_subplot(4, 2, 8)
        ax8.plot(data.index.values, data['pt2_net_a'], 'bo')
        ax8.set_ylabel('Net accel (cm/s2)', color='b')
        ax8.tick_params('y', colors='b')

        plt.show()

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

def extract_data():
    """Extracts fish position calculates velocities & acceleration.

    This function extracts data from a directory of one or more
    .xypts.csv files, assuming they track the snout of a fish (pt 1)
    and the tail of the fish (pt 2) from a lateral (vid 1) and a ventral
    (vid 2) view. Trial information is taken from a trial_info.csv file
    in the same directory. Position data is prepared for differentiation
    to velocity using a Savitsky-Golay filter, and velocity is also
    smoothed for differentiation to accelerations. All of these,
    and other trial information are returned as a dictionary tracklist,
    with each key representing one trial. Each key accesses the data and
    trial info for that trial.

    Args:
        None (hardcoded)

    Returns:
        tracklist['trial'] (dict): each key referring to its own
        dictionary with the following entries:
            tracklist['trial']['sequence'] (str): name of trial
            tracklist['trial']['fish'] (str): name of fish
            tracklist['trial']['species'] (str): name of species
            tracklist['trial']['fish_TL'](float): fish total length
            tracklist['trial']['FPS'](int): frame rate in frames/second
            tracklist['trial']['behavior'](str): behavior as coded by
                                                filmer.
                                                'A': linear acc.
                                                'B': burst acc.
                                                'S': steady swim
            tracklist['trial']['data'] (DataFrame): trial data, w/ the
                                             entries as defined inline

    """
    # Useful snippet to pull species from Fish later
    def letters(input):
        valids = ""
        for character in input:
            if character.isalpha():
                valids += character
        return valids

    # YOU MUST SET THIS FOR YOUR OWN DATA
    trial_info = pd.read_csv(
        r'C:\Users\Kara\PycharmProjects\FishTrackProcessing\Data\Trial_info.csv',
        sep=',')
    trial_info = trial_info.set_index('Trial_name')

    # YOU MUST SET THIS FOR YOUR OWN DATA
    folder = r'C:\Users\Kara\PycharmProjects\FishTrackProcessing\Data'

    #  Uncomment for User Input
    # folder = filedialog.askdirectory()  #Ask user for directory
    # framerate = float(input('Enter frame rate in frames per second:'))

    # Initialize a list of dictionaries to contain each trial's data
    tracklist = {}
    count = 0  # Initialize the count

    def my_round(x):
        return round(x * 4) / 4

    for filename in os.listdir(
            folder):  # For all files in the directory
        if filename.endswith("xypts.csv"):  # that end with 'xypts.csv'

            # Extract info from filename
            filepath = folder + '/' + filename
            file_info = filename.split("_")
            fish = file_info[0]
            species = letters(fish)
            sequence = file_info[1]
            trial_name = fish + sequence
            framerate = trial_info['FPS'][trial_name]
            L_calib = trial_info['ScaleL_cm/px'][trial_name]
            V_calib = trial_info['ScaleV_cm/px'][trial_name]
            init_Speed = trial_info['InitialSpd_cm'][trial_name]
            init_Speed_L = my_round(trial_info['InitialSpd_BLs'][
                                        trial_name])
            fish_TL = trial_info['Fish_TL_cm'][trial_name]
            behavior = trial_info['Behavior'][trial_name]

            df = pd.read_csv(filepath, sep=',')
            df = df.rename(
                columns={'pt1_cam1_Y': 'pt1z', 'pt1_cam2_X': 'pt1x',
                         'pt1_cam2_Y': 'pt1y', 'pt2_cam1_Y': 'pt2z',
                         'pt2_cam2_X': 'pt2x', 'pt2_cam2_Y': 'pt2y'})

            # Convert position to cm
            df['pt1z'] = df['pt1z'] * L_calib
            df['pt1x'] = df['pt1x'] * V_calib
            df['pt1y'] = df['pt1y'] * V_calib
            df['pt2z'] = df['pt2z'] * L_calib
            df['pt2x'] = df['pt2x'] * V_calib
            df['pt2y'] = df['pt2y'] * V_calib

            # Generate time array
            df['time'] = np.linspace(0, len(df['pt1x']) * (
            1.0 / framerate),
                                     num=len(df['pt1x']),
                                     endpoint=False)
            df = df.set_index(['time'])

            # Smooth position data using savitzky golay
            df['pt1x_smth'] = scipy.signal.savgol_filter(
                df['pt1x'], window_length=121, polyorder=2)
            df['pt1y_smth'] = scipy.signal.savgol_filter(
                df['pt1y'], window_length=121, polyorder=2)
            df['pt1z_smth'] = scipy.signal.savgol_filter(
                df['pt1z'], window_length=121, polyorder=2)

            df['pt2x_smth'] = scipy.signal.savgol_filter(
                df['pt2x'], window_length=121, polyorder=2)
            df['pt2y_smth'] = scipy.signal.savgol_filter(
                df['pt2y'], window_length=121, polyorder=2)
            df['pt2z_smth'] = scipy.signal.savgol_filter(
                df['pt2z'], window_length=121, polyorder=2)

            # Calculate First Discrete Differences (Velocity)
            cols_to_use1 = ['pt1x_smth', 'pt1y_smth', 'pt1z_smth',
                            'pt2x_smth',
                            'pt2y_smth', 'pt2z_smth']
            df2 = df.loc[:, cols_to_use1].diff()
            df2 = df2.rename(columns={
                'pt1z_smth': 'pt1z_v', 'pt1x_smth': 'pt1x_v',
                'pt1y_smth': 'pt1y_v', 'pt2z_smth': 'pt2z_v',
                'pt2x_smth': 'pt2x_v', 'pt2y_smth': 'pt2y_v'})

            # Making forward, up positive
            df2['pt1x_v'] = -df2['pt1x_v']
            df2['pt1y_v'] = -df2['pt1y_v']
            df2['pt1z_v'] = -df2['pt1z_v']
            df2['pt2x_v'] = -df2['pt2x_v']
            df2['pt2y_v'] = -df2['pt2y_v']
            df2['pt2z_v'] = -df2['pt2z_v']

            # Add initial x-velocity
            df2['pt1x_v'] = df2['pt1x_v'].add(init_Speed)
            df2['pt2x_v'] = df2['pt2x_v'].add(init_Speed)

            # Smooth velocity data using savitzky golay
            df2['pt1x_v_smth'] = scipy.signal.savgol_filter(
                df2['pt1x_v'], window_length=121, polyorder=3)
            df2['pt1y_v_smth'] = scipy.signal.savgol_filter(
                df2['pt1y_v'], window_length=121, polyorder=3)
            df2['pt1z_v_smth'] = scipy.signal.savgol_filter(
                df2['pt1z_v'], window_length=121, polyorder=3)

            df2['pt2x_v_smth'] = scipy.signal.savgol_filter(
                df2['pt2x_v'], window_length=121, polyorder=3)
            df2['pt2y_v_smth'] = scipy.signal.savgol_filter(
                df2['pt2y_v'], window_length=121, polyorder=3)
            df2['pt2z_v_smth'] = scipy.signal.savgol_filter(
                df2['pt2z_v'], window_length=121, polyorder=3)

            # Calculate Second Discrete Differences (Acceleration)
            cols_to_use2 = ['pt1x_v_smth', 'pt1y_v_smth', 'pt1z_v_smth',
                            'pt2x_v_smth', 'pt2y_v_smth', 'pt2z_v_smth']
            df3 = df2.loc[:, cols_to_use2].diff()
            df3 = df3.rename(columns={
                'pt1z_v_smth': 'pt1z_a', 'pt1x_v_smth': 'pt1x_a',
                'pt1y_v_smth': 'pt1y_a', 'pt2z_v_smth': 'pt2z_a',
                'pt2x_v_smth': 'pt2x_a', 'pt2y_v_smth': 'pt2y_a'})

            # Merge all this shit
            df = df.merge(df2, how='outer', right_index=True,
                          left_index=True)
            df = df.merge(df3, how='outer', left_index=True,
                          right_index=True)

            # Calculate net velocity and net accel
            df['pt1_net_v'] = np.sqrt(df['pt1x_v_smth']**2 + df[
                'pt1y_v_smth']**2 + df['pt1z_v_smth']**2)
            df['pt2_net_v'] = np.sqrt(df['pt2x_v_smth'] ** 2 + df[
                'pt2y_v_smth'] ** 2 + df['pt2z_v_smth'] ** 2)



            df['pt1_net_a']= df['pt1_net_v'].diff()
            df['pt2_net_a'] = df['pt2_net_v'].diff()

            # Put all of these into the appropriate object in tracklist
            tracklist[trial_name] = {'sequence': trial_name,
                                     'fish': fish,
                                     'fish_TL': fish_TL,
                                     'start_spd': init_Speed,
                                     'start_spd_BLs': init_Speed_L,
                                     'species': species,
                                     'FPS': framerate,
                                     'behavior': behavior,
                                     'data': df}
            # Advance the count
            count = count + 1
    return tracklist

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
                             Also: Alternatively, to iterate
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

def make_subset(group_by1, identifier1, tracklist, group_by2=None,
                identifier2=None, group_by3=None, identifier3=None):
    """Generates lists suitable for use as subsets in other fns

    This function produces lists of trials from tracklist grouped by
    factors of interest, that are suitable for use in other
    functions, including finbeat_calc(), plot_track(), plot_accel(),
    check_plots(), and plot_analysis() in particular. Facilitates
    easy subset creation by species or fish. Possible identifiers are 
    any of the string, float, or integer associated keys of a 
    tracklist: ['species', 'fish', 'behavior', 'fish_TL', 
                'start_spd','start_spd_BLs']

    Args:
        group_by1 (str): the factor to search for your identifier of
                        interest, usually 'fish' or 'species' or 
                        'behavior'
        identifier1 (str): the specific identifier you want to index,
                           e.g. 'Bass' or 'BTrout' if group_by =
                          'species'; or 'Bass1' or 'BTrout2' if
                          group_by = 'fish'.
        tracklist (dict): the compiled position, velocity,
                          and acceleration data for all trials
                           produced by extract_data()
        group_by2 (str): optional, as group_by1
        identifier2(str): req. if group_by2 is used, as identifier1. 
        group_by3 (str): optional, as group_by1. Use group_by2 first. 
        identifier3 (str): req. if group_by3 is used, as identifier1.

    Returns:
        subset (list): a list of strings containing the trial names
                        of trials matching the specified identifier,
                        which can be used in other functions.
    """
    subset = []

    for i in tracklist.keys():
        if tracklist[i][group_by1] == identifier1:
            subset.append(i)

    if group_by2 != None:
        subset2 = list(subset)
        for j in subset:
            if tracklist[j][group_by2] != identifier2:
                subset2.remove(j)
    else:
        return subset

    if group_by3 != None:
        subset3 = list(subset2)
        for k in subset2:
            if tracklist[k][group_by3] != identifier3:
                subset3.remove(k)
    else:
        return subset2

    return subset3

def mult_reg_boot(formula, y, df, reps=2000):
    """Runs bootstrap reps of multiple linear regression to est outputs

           Args:
                formula (str): string identifying the formula as 
                        input to stats models, typically of the form
                         'Y ~ X1 * X2' , or something like that. 
                         See the Jupyter notebook for examples. 
               y (string): string identifying the Pandas Series within 
                        df that contains the response variable
               df (Pandas DataFrame): one of the dataframes prepared 
                        using script finbeat_params_prep.py
               reps (int): number of bootstrap replicates

           Returns:
    """
    # Make a list to hold model objects
    # Method 1: Resampling residuals

    boot_outputs_method1 = []

    # Method 1
    reg_model1 = smf.ols(formula=formula, data=df).fit()
    predicted = reg_model1.fittedvalues
    residuals = reg_model1.resid
    stu_resid = reg_model1.wresid
    pearson_resid = reg_model1.resid_pearson

    for rep in np.arange(1, reps):
        df_copy = df.copy()
        # randomly resample residuals and add these random resids to y
        random_resid = np.random.choice(residuals, size=len(
            residuals), replace=True)
        df_copy[y] = df_copy[y] + random_resid
        # refit model using fake y and store output
        reg_model1_rep = smf.ols(formula=formula, data=df_copy).fit()
        boot_outputs_method1.append(reg_model1_rep)

    return boot_outputs_method1

def plot_accel(tracklist_subset, tracklist):
    """Plot tail beat motion and snout streamwise acceleration 

    Does what it says above, using data from tracklist.

            Args:
                tracklist_subset (list): a subset of trials, typically 
                    generated using the convenience function 
                    make_subset()
                tracklist (dict): a tracklist produced by extract_data()

           """

    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)
        data = tracklist[trial]['data']
        # filename = str(trial) + '.pdf'

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist[trial]['behavior'])
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(data.index.values, data['pt2y_smth'], 'bo')
        ax1.set_ylabel('Tail Excursion (cm)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(data.index.values, data['pt1x_a'], 'r.')
        ax2.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax1.set_xlabel('Time (s)')
        ax2.tick_params('y', colors='r')
        plt.axhline(0, color='r', linewidth=4, linestyle='dashed')
        # plt.savefig(filename)
        plt.show()

def plot_accels_forfig(plotnum, trial, tracklist):
    """Plot tail beat acceleration and snout streamwise acceleration 
       This function is used within a figure script to produce panels 
       for a composite figure. See "Fig6_Trial Traj.py"
       Does what it says above, using data from tracklist. Tail tip 
       accelerations are presented as absolute values. 

               Args:
                   plotnum (int): the number indicating the position 
                   of the panel using GridSpec conventions
                   trial(str): a trial name
                   tracklist (dict): tracklist produced by 
                        extract_data()

                Returns: 
                    ax, ax2 (matplotlib axes): the panel showing 
                    acceleration data

              """
    ax = fig6.add_subplot(plotnum)
    ax.plot(tracklist[trial]['data'].index.values,
            abs(tracklist[trial]['data']['pt2_net_a']), c='blue')
    ax2 = ax.twinx()
    ax2.plot(tracklist[trial]['data'].index.values,
             tracklist[trial]['data']['pt1_net_a'], c='orange')
    ax.set_ylim(tracklist[trial]['data']['pt1_net_a'].min(),
                abs(tracklist[trial]['data']['pt2_net_a'].max()))
    ax2.set_ylim(tracklist[trial]['data']['pt1_net_a'].min(),
                 abs(tracklist[trial]['data']['pt1_net_a'].max()))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tailtip Acceleration (cm$^2$)')
    ax.yaxis.label.set_color('blue')
    ax2.set_ylabel('Snout Acceleration (cm$^2$)')
    ax2.yaxis.label.set_color('orange')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 2))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3, 2))
    return ax, ax2

def plot_analysis(subset_name, finbeats_subset, finbeat_data,
                    tracklist, zaxis='A', lines=True, cutoff=False,
                    save=False):
    """Plots finbeats in (period, amplitude, acceleration) space.

    This function takes finbeat data from a specified output of
    finbeat_calc() and plots each individual finbeat in (period,
    amplitude, maximum [acceleration or velocity]) space. The 
    finbeat_data argument specifies whether the finbeats to be 
    plotted come from peak-to-peak or trough-to-trough calculations. 
    The maximum acceleration is the maximum acceleration between the 
    finbeat start and finbeat end times. The number of total 
    finbeats is printed at the end.

    Args:
        subset_name (string): some string identifying what's in your
                                subset, to be used as the plot title
        finbeats_subset (list): a list of strings with the trial
                            names of the desired trials from finbeats.
                             Note: The list (even of a single
                             element) must be contained in square
                             brackets. You'll probably want to use
                             the subset generating function:
                             make_subset()
        finbeat_data (dict): use either finbeat_byP to do analysis
                            on finbeats as defined by peaks first,
                            or finbeat_byT to use finbeats defined by
                            troughs first. These must be created
                            beforehand by the function finbeat_calc()
        zaxis (str): must be of value "A" or "V". Indicates whether to plot
                        acceleration or velocity.
        tracklist (dict): the compiled position, velocity,
                          and acceleration data for all trials
                           produced by extract_data()
        lines (Bool): if True, adds lines up from x-y plane to z_value
        cutoff (Bool): if True, cuts off z axis at hard-coded maximum value
        save (Bool): if True, saves to svg instead of printing to screen

    Returns:
        Nothing
    """
    count_n = 0  # start counting finbeats

    # find max initial speed for coloring by speed
    speeds = []
    for trial in finbeats_subset:
        speeds.append(tracklist[trial]['start_spd'])
    max_spd = max(speeds)

    # find x and y max and min axis limits
    x_vals = []
    y_vals = []
    z_vals = []
    for trial in finbeats_subset:
        for finbeat in finbeat_data[trial].index.values:
            x_vals.append(finbeat_data[trial]['period'][finbeat])
            y_vals.append(finbeat_data[trial]['amplitude'][finbeat])

    x_max = np.nanmax(x_vals)
    y_max = np.nanmax(y_vals)

    # Pull a colormap
    cm = plt.get_cmap("plasma")

    # Set up the figure and choose an appropriate z-axis label
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    fig.suptitle(subset_name, position=(0.5,0.9))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Amplitude (cm)')
    if zaxis == 'V':
        ax1.set_zlabel('\nMaximum Inst. Velocity(cm/s)')
    else:
        ax1.set_zlabel('\nMaximum Accel (cm/s2)')
    ax1.set_xlim3d(0, x_max)
    ax1.set_ylim3d(0, y_max)

    # for each trial of interest
    for trial in finbeats_subset:
        # for each finbeat within that trial
        for finbeat in finbeat_data[trial].index.values:
            # get the period
            # period_mask = finbeat_data[trial]['period'].loc[finbeat]
            period = finbeat_data[trial]['period'][finbeat]

            # get the amplitude
            # amplitude_mask = finbeat_data[trial]['amplitude'].loc[
            # finbeat]
            amplitude = finbeat_data[trial]['amplitude'][finbeat]

            # get the start time
            # start_mask = finbeat_data[trial]['time'].loc[finbeat]
            start = finbeat_data[trial]['time'][finbeat]
            # get the end time
            # end_mask = finbeat_data[trial]['endtime'].loc[finbeat]
            end = finbeat_data[trial]['endtime'][finbeat]

            # find the maximum acceleration or velocity in that time range
            if zaxis == 'A':
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_a'][start:end].max()
                z_vals.append(zcolumn)
            elif zaxis == 'V':
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_v'][start:end].max()
                z_vals.append(zcolumn)
            else:  # If they fuck up, make it acceleration
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_a'][start:end].max()
                z_vals.append(zcolumn)

            # pull the initial speed and behavior
            init_spd = tracklist[trial]['start_spd']
            behavior_type = tracklist[trial]['behavior']
            if behavior_type == 'B':
                behavior = '*'
            elif behavior_type == 'A':
                behavior = '+'
            else:
                behavior = 'o'

            # add the point
            if cutoff == True and zaxis == 'A':
                z_max = 0.00005
            else:
                z_max = np.nanmax(z_vals)
            if zcolumn <= z_max and lines == True and zcolumn >= 0:
                p = ax1.plot(xs=[period, period], ys=[amplitude, amplitude],
                             zs=[0, zcolumn],
                             linestyle='solid', c=cm(init_spd / max_spd),
                             alpha=0.8, linewidth=0.5)
                p = ax1.scatter3D(xs=period,
                                  ys=amplitude,
                                  zs=zcolumn,
                                  zdir='z', s=20, marker=behavior,
                                  c=init_spd,
                                  cmap=cm, edgecolor='none', vmin=0,
                                  vmax=max_spd)
                count_n += 1

    ax1.set_zlim3d(0, z_max)
    cbar = plt.colorbar(p,shrink=0.7, pad = 0.1)
    cbar.set_label('Initial Speed (cm/s)', rotation=270, labelpad=10)
    if save == True:
        plt.savefig(str(subset_name)+".svg", format="svg")
    else:
        plt.show()
    print(count_n)

def plot_analysis_forfig(rows, columns, number, finbeats_subset,
                         finbeat_data, tracklist, zaxis='A', lines=True,
                         cutoff=False):
    """Plots finbeats in (period, amplitude, acceleration) space.

    This function takes finbeat data from a specified output of
    finbeat_calc(), and plots each individual finbeat in (period,
    amplitude, maximum acceleration (or velocity) space. The 
    finbeat_data argument specifies whether the finbeats to be 
    plotted come from peak-to-peak or trough-to-trough calculations. 
    The maximum acceleration is the maximum acceleration between the 
    finbeat start and finbeat end times. 

    Note: This figure is used to make subpanels for composite 
    figures. See Figure 2 from the paper.

    Args:
        rows (int): Number of rows in composite figure
        columns (int): Number columns in composite figure
        number (int): Number of this subplot
        finbeats_subset (list): a list of strings with the trial
                            names of the desired trials from finbeats.
                             Note: The list (even of a single
                             element) must be contained in square
                             brackets. You'll probably want to use
                             the subset generating function:
                             make_subset()
        finbeat_data (dict): use either finbeat_byP to do analysis
                            on finbeats as defined by peaks first,
                            or finbeat_byT to use finbeats defined by
                            troughs first. These must be created
                            beforehand by the function finbeat_calc()
        zaxis (str): must be of value "A" or "V". Indicates whether to plot
                        acceleration or velocity.
        tracklist (dict): the compiled position, velocity,
                          and acceleration data for all trials
                           produced by extract_data()
        lines (Bool): if True, adds lines up from x-y plane to z_value
        cutoff (Bool): if True, cuts off z axis at hard-coded maximum value

    Returns:
        ax1 (matplotlib Axes3D object)

    """
    speeds_cb = [0] * len(tracklist.keys())
    count_cb = 0
    for i in tracklist.keys():
        speeds_cb[count_cb] = tracklist[i]['start_spd']
        count_cb += 1
    speed_cb = max(speeds_cb)

    count_n = 0  # start counting finbeats

    # find max initial speed for coloring by speed
    speeds = []
    for trial in finbeats_subset:
        speeds.append(tracklist[trial]['start_spd'])
    max_spd = max(speeds)

    # find x and y max and min axis limits
    x_vals = []
    y_vals = []
    z_vals = []
    for trial in finbeats_subset:
        for finbeat in finbeat_data[trial].index.values:
            x_vals.append(finbeat_data[trial]['period'][finbeat])
            y_vals.append(finbeat_data[trial]['amplitude'][finbeat])

    x_max = np.nanmax(x_vals)
    y_max = np.nanmax(y_vals)

    # Pull a colormap
    cm = plt.get_cmap("plasma")

    # Set up the figure and choose an appropriate z-axis label
    ax1 = fig2.add_subplot(rows, columns, number, projection='3d')
    ax1.set_xlabel('Period (s)')
    ax1.set_ylabel('Amplitude (cm)')
    if zaxis == 'V':
        ax1.set_zlabel('\nMax. Inst. Velocity (cm/s)')
    else:
        ax1.set_zlabel('\nMax. Acceleration (cm/s $^2$)')
    ax1.set_xlim3d(0, x_max)
    ax1.set_ylim3d(0, y_max)

    # for each trial of interest
    for trial in finbeats_subset:
        # for each finbeat within that trial
        for finbeat in finbeat_data[trial].index.values:
            # get the period
            # period_mask = finbeat_data[trial]['period'].loc[finbeat]
            period = finbeat_data[trial]['period'][finbeat]

            # get the amplitude
            # amplitude_mask = finbeat_data[trial]['amplitude'].loc[
            # finbeat]
            amplitude = finbeat_data[trial]['amplitude'][finbeat]

            # get the start time
            # start_mask = finbeat_data[trial]['time'].loc[finbeat]
            start = finbeat_data[trial]['time'][finbeat]
            # get the end time
            # end_mask = finbeat_data[trial]['endtime'].loc[finbeat]
            end = finbeat_data[trial]['endtime'][finbeat]

            # find the maximum acceleration or velocity in that time range
            if zaxis == 'A':
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_a'][start:end].max()
                z_vals.append(zcolumn)
            elif zaxis == 'V':
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_v'][start:end].max()
                z_vals.append(zcolumn)
            else:  # If they fuck up, make it acceleration
                zcolumn = tracklist[trial]['data'][
                              'pt1_net_a'][start:end].max()
                z_vals.append(zcolumn)

            # pull the initial speed and behavior
            init_spd = tracklist[trial]['start_spd']
            behavior_type = tracklist[trial]['behavior']
            if behavior_type == 'B':
                behavior = '*'
                size = 60
            elif behavior_type == 'A':
                behavior = 'P'
                size = 50
            else:
                behavior = 'o'
                size = 30

            # add the point
            if cutoff == True and zaxis == 'A':
                z_max = 0.00005
            else:
                z_max = np.nanmax(z_vals)
            if zcolumn <= z_max and lines == True and zcolumn >= 0:
                p = ax1.plot(xs=[period, period],
                             ys=[amplitude, amplitude],
                             zs=[0, zcolumn],
                             linestyle='solid',
                             c=cm(init_spd / max_spd),
                             alpha=0.8, linewidth=0.5)
                p = ax1.scatter3D(xs=period,
                                  ys=amplitude,
                                  zs=zcolumn,
                                  zdir='z', s=size, marker=behavior,
                                  c=init_spd,
                                  cmap=cm, edgecolor='none', vmin=0,
                                  vmax=speed_cb)
                count_n += 1

    ax1.set_zlim3d(0, z_max)
    pane_gray = 1.0
    ax1.w_xaxis.set_pane_color((pane_gray, pane_gray, pane_gray, 1.0))
    ax1.w_yaxis.set_pane_color((pane_gray, pane_gray, pane_gray, 1.0))
    ax1.w_zaxis.set_pane_color((pane_gray, pane_gray, pane_gray, 1.0))
    # cbar = plt.colorbar(p,shrink=0.7, pad = 0.1)
    # cbar.set_label('Initial Speed (cm/s)', rotation=270, labelpad=10)
    # if save == True:
    # plt.savefig(str(subset_name)+".svg", format="svg")
    # else:
    # plt.show()
    # print(count_n)

    return ax1

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

def plot_finbeats(tracklist_subset, tracklist, finbeat_params):
    """Don't use this, it's deprecated
 
         Args:
             tracklist_subset (list): a subset of trials, typically 
                     generated using the convenience function 
                     make_subset()
             tracklist (dict): a tracklist produced by extract_data()
             finbeat_params (dataframe): use either finbeat_byP to do 
                     analysis on finbeats as defined by peaks first,
                     or finbeat_byT to use finbeats defined by
                     troughs first. These must be created beforehand by
                     the function finbeat_calc()
    """
    for trial in tracklist_subset:
        # shorten references for finbeat variables
        fb_peaktimes = finbeat_params[trial]['finbeat_peak_times']
        fb_effort = finbeat_params[trial]['finbeat_effort']
        fb_amplitude = finbeat_params[trial]['finbeat_amplitudes']
        fb_period = finbeat_params[trial]['finbeat_periods']

        fig = plt.figure()
        fig.suptitle(tracklist[trial]['sequence'])

        # Subplot 1: Effort and X-Acceleration on Time
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(fb_peaktimes, fb_effort, 'bo')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Tailbeat Effort (cm/s)', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax2.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax2.tick_params('y', colors='r')

        # Sublplot 2: Effort and X-Velocity on Time
        ax3 = fig.add_subplot(3, 2, 2)
        ax3.plot(fb_peaktimes, fb_effort, 'bo')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tailbeat Effort (cm/s)', color='b')
        ax3.tick_params('y', colors='b')
        ax4 = ax3.twinx()
        ax4.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax4.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax4.tick_params('y', colors='r')

        # Subplot 3: Amplitude and X-Acceleration on Time
        ax5 = fig.add_subplot(3, 2, 3)
        ax5.plot(fb_peaktimes, fb_amplitude, 'bo')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Tailbeat Amplitude (cm)', color='b')
        ax5.tick_params('y', colors='b')
        ax6 = ax5.twinx()
        ax6.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax6.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax6.tick_params('y', colors='r')

        # Subplot 4: Amplitude and X-Velocity on Time
        ax7 = fig.add_subplot(3, 2, 4)
        ax7.plot(fb_peaktimes, fb_amplitude, 'bo')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Tailbeat Amplitude (cm)', color='b')
        ax7.tick_params('y', colors='b')
        ax8 = ax7.twinx()
        ax8.plot(tracklist[trial]['data'].index.values,
                 -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax8.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax8.tick_params('y', colors='r')

        # Subplot 5: Period and X-Acceleration on Time
        ax9 = fig.add_subplot(3, 2, 5)
        ax9.plot(fb_peaktimes, fb_period, 'bo')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Tailbeat Period (s)', color='b')
        ax9.tick_params('y', colors='b')
        ax10 = ax9.twinx()
        ax10.plot(tracklist[trial]['data'].index.values,
                  -tracklist[trial]['data']['pt1x_a'], 'r.')
        ax10.set_ylabel('Streamwise accel (cm/s2)', color='r')
        ax10.tick_params('y', colors='r')

        # Subplot 6: Period and X-Velocity on Time
        ax11 = fig.add_subplot(3, 2, 6)
        ax11.plot(fb_peaktimes, fb_period, 'bo')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Tailbeat Period (s)', color='b')
        ax11.tick_params('y', colors='b')
        ax12 = ax11.twinx()
        ax12.plot(tracklist[trial]['data'].index.values,
                  -tracklist[trial]['data']['pt1x_v'], 'r.')
        ax12.set_ylabel('Streamwise Velocity(cm/s)', color='r')
        ax12.tick_params('y', colors='r')
        plt.show()

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
        raw_data = raw_data - base
        time = raw_data.index.values

        amp = sin_corr_df['Est.Amplitude'][trial]
        freq = sin_corr_df['Est.Freq'][trial]
        phase = sin_corr_df['Est.Phase'][trial]
        offset = sin_corr_df['Est.Offset'][trial]
        cor_coeff = str(np.round(sin_corr_df['Pearsons'][trial], 2))
        pvalue = str(np.round(sin_corr_df['Pvalue'][trial], 3))
        annotation = str('r = ' + cor_coeff + '\np = ' + pvalue)

        data_fit = amp * np.sin(
            2 * np.pi * freq * time + phase) + offset

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle(
            trial + ' ' + behavior + ' ' + str(init_speed) + 'BL/s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (cm)')
        ax.plot(time, raw_data, label='Raw Data')
        ax.plot(time, data_fit, label="Sine Wave Fit")
        ax.text(0.02, 0.98, annotation, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)
        ax.legend()
        plt.show()

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

def plot_trace_forfig(rows, columns, number, trial, tracklist):
    """Plots detrended tailtip motion against time.

        Does what it says above. Intended for use to create panels 
        for composite figures. See Figure 5 for example. 

        Args:
            rows (int): Number of rows in composite figure
            columns (int): Number columns in composite figure
            number (int): Number of this subplot
            trial(string): the trial name
            tracklist (dict): the compiled position, velocity,
                              and acceleration data for all trials
                               produced by extract_data()

        Returns:
            ax (matplotlib Axes)
        """

    raw_data = tracklist[trial]['data']['pt2y']
    behavior = tracklist[trial]['behavior']
    fish = tracklist[trial]['fish']
    init_speed = tracklist[trial]['start_spd_BLs']
    base = peakutils.baseline(raw_data, 3)  # Find bkgrd trend
    raw_data = raw_data - base
    time = raw_data.index.values

    if fish == 'Bass1':
        col = 'darkblue'
    else:
        col = 'darkred'

    ax = fig3.add_subplot(rows, columns, number)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tail Tip Position (cm)')
    ax.plot(time, raw_data, col)

    return ax

def plot_track(tracklist_subset, tracklist):
    """Plot the 3D position, streamwise velocity, and streamwise accel

     Plot the 3D position, streamwise velocity, and streamwise accel
     of the snout and the tail tip from data output from 
     extract_data(). Produces a multipanel plot with position in 3D 
     for pts 1 and 2, and derived values in 2D (against time). 

    Args:
       tracklist_subset : 1D array_like
           List of strings indicating sequence names of desired trials.         
       tracklist : pandas dataframe
           tracklist dataframe produced by process_points.py

    Returns:
        None
       """

    for trial in tracklist_subset:  # Iterates over all available trials

        # Scale time for colormap
        scaled_time = (tracklist[trial]['data'].index.values -
                       tracklist[trial]['data'].index.values.min()) / \
                      tracklist[trial]['data'].index.values.ptp()
        timemax = max(tracklist[trial]['data'].index.values)

        # Determining axis limits
        pt1max_v = tracklist[trial]['data']['pt1x_v_smth'].max()
        pt1min_v = tracklist[trial]['data']['pt1x_v_smth'].min()
        pt1max_a = tracklist[trial]['data']['pt1x_a'].max()
        pt1min_a = tracklist[trial]['data']['pt1x_a'].min()
        pt1_vbuff = (pt1max_v - pt1min_v) * 0.05  # Adds margin of 5%
        pt1_abuff = (pt1max_a - pt1min_a) * 0.05
        pt1_vmaxlim = pt1max_v + pt1_vbuff
        pt1_vminlim = pt1min_v - pt1_vbuff
        pt1_amaxlim = pt1max_a + pt1_abuff
        pt1_aminlim = pt1min_a - pt1_abuff

        pt2max_v = tracklist[trial]['data']['pt2x_v_smth'].max()
        pt2min_v = tracklist[trial]['data']['pt2x_v_smth'].min()
        pt2max_a = tracklist[trial]['data']['pt2x_a'].max()
        pt2min_a = tracklist[trial]['data']['pt2x_a'].min()
        pt2_vbuff = (pt2max_v - pt2min_v) * 0.05
        pt2_abuff = (pt2max_a - pt2min_a) * 0.05
        pt2_vmaxlim = pt2max_v + pt2_vbuff
        pt2_vminlim = pt2min_v - pt2_vbuff
        pt2_amaxlim = pt2max_a + pt2_abuff
        pt2_aminlim = pt2min_a - pt2_abuff

        # Pull from colormap (here cubehelix)
        colors = plt.cm.cubehelix(scaled_time)

        # Raw Plot Pt 1, 3D
        fig = plt.figure()
        fig.set_figheight(20)
        fig.set_figwidth(15)
        fig.suptitle(tracklist[trial]['sequence'] + ' ' +
                     tracklist[trial]['behavior'])
        ax1 = fig.add_subplot(4, 2, 1, projection='3d')
        ax1.set_title('Pt 1 Raw Position')
        ax1.scatter3D(xs=tracklist[trial]['data']['pt1x'],
                      ys=tracklist[trial]['data']['pt1y'],
                      zs=tracklist[trial]['data']['pt1z'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax1.autoscale(enable=True, tight=True)
        ax1.set_xlabel('X position')
        ax1.set_ylabel('Y position')
        ax1.set_zlabel('Z position')
        m = cm.ScalarMappable(cmap=cm.cubehelix)
        m.set_array(tracklist[trial]['data'].index.values)
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Raw Plot Pt 2, 3D
        ax2 = fig.add_subplot(4, 2, 2, projection='3d')
        ax2.set_title('Pt 2 Raw Position')
        ax2.scatter3D(xs=tracklist[trial]['data']['pt2x'],
                      ys=tracklist[trial]['data']['pt2y'],
                      zs=tracklist[trial]['data']['pt2z'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # 3D Scatter plot
        ax2.autoscale(enable=True, tight=True)
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Y position')
        ax2.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Smoothed Data Pt 1, 3D
        ax3 = fig.add_subplot(4, 2, 3, projection='3d')
        ax3.set_title('Pt 1 Smoothed Position')
        ax3.scatter3D(xs=tracklist[trial]['data']['pt1x_smth'],
                      ys=tracklist[trial]['data']['pt1y_smth'],
                      zs=tracklist[trial]['data']['pt1z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # Scatter plot
        ax3.autoscale(enable=True, tight=True)
        ax3.set_xlabel('X position')
        ax3.set_ylabel('Y position')
        ax3.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Smoothed Data Pt 2, 3D
        ax4 = fig.add_subplot(4, 2, 4, projection='3d')
        ax4.set_title('Pt 2 Smoothed Position')
        ax4.scatter3D(xs=tracklist[trial]['data']['pt2x_smth'],
                      ys=tracklist[trial]['data']['pt2y_smth'],
                      zs=tracklist[trial]['data']['pt2z_smth'],
                      zdir='z', s=3, c=colors, marker='o',
                      edgecolor='none')  # Scatter plot
        ax4.autoscale(enable=True, tight=True)
        ax4.set_xlabel('X position')
        ax4.set_ylabel('Y position')
        ax4.set_zlabel('Z position')
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Velocity Pt 1
        ax5 = fig.add_subplot(4, 2, 5)
        ax5.set_title('Pt 1 Streamwise Velocity')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt1x_v_smth'],
                    c=colors, edgecolor='none')
        ax5.set_xlim([0, timemax])
        ax5.set_ylim([pt1_vminlim, pt1_vmaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Velocity Pt 2
        ax6 = fig.add_subplot(4, 2, 6)
        ax6.set_title('Pt 2 Streamwise Velocity')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt2x_v_smth'],
                    c=colors, edgecolor='none')
        ax6.set_xlim([0, timemax])
        ax6.set_ylim([pt2_vminlim, pt2_vmaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Accel Pt 1
        ax7 = fig.add_subplot(4, 2, 7)
        ax7.set_title('Pt 1 Streamwise Acceleration')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt1x_a'],
                    c=colors, edgecolor='none')
        ax7.set_xlim([0, timemax])
        ax7.set_ylim([pt1_aminlim, pt1_amaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)

        # Streamwise Accel Pt 2
        ax8 = fig.add_subplot(4, 2, 8)
        ax8.set_title('Pt 2 Streamwise Acceleration')
        plt.scatter(x=tracklist[trial]['data'].index.values,
                    y=tracklist[trial]['data']['pt2x_a'],
                    c=colors, edgecolor='none')
        ax8.set_xlim([0, timemax])
        ax8.set_ylim([pt2_aminlim, pt2_amaxlim])
        plt.colorbar(m, shrink=0.5, aspect=10)
        plt.show()

def plot_traj(trial, tracklist, finbeat_subset):
    """Plots a single trial's finbeats against steady finbeats 

    Plots a single trial's finbeats against steady finbeats for the 
    same species at the same initial speed. 

       Args:
           trial (string): some string identifying what's in your
                                   subset, to be used as the plot title
           finbeat_subset (): 
           tracklist (dict): the compiled position, velocity,
                             and acceleration data for all trials
                              produced by extract_data()

       Returns:
           fig (matplotlib figure)
       """

    # Get info for focal trial
    behavior = tracklist[trial]['behavior']
    species = tracklist[trial]['species']
    speed = tracklist[trial]['start_spd_BLs']

    # Make the corresponding subset of steady trials
    steady_subset = make_subset(group_by1='species',
                                identifier1=species,
                                tracklist=tracklist,
                                group_by2='behavior',
                                identifier2='S',
                                group_by3='start_spd_BLs',
                                identifier3=speed)

    # Get all steady finbeats for steady subset trials
    steady_finbeats = []
    # Make (speed, period, amplitude) tuples
    for i in steady_subset:
        for j in finbeat_subset[i].index.values:
            start = finbeat_subset[i]['time'][j]
            stop = finbeat_subset[i]['endtime'][j]
            speed = tracklist[i]['data'][
                        'pt1_net_v'][start:stop].max()
            steady_finbeats.append((speed,
                                    finbeat_subset[i]['period'][j],
                                    finbeat_subset[i]['amplitude'][j]))

    # Generate tuples for the focal trial with finbeat parameters)
    trial_finbeats = []
    count = 0
    for k in finbeat_subset[trial].index.values:
        start = finbeat_subset[trial]['time'][k]
        stop = finbeat_subset[trial]['endtime'][k]
        period = finbeat_subset[trial]['period'][k]
        amplitude = finbeat_subset[trial]['amplitude'][k]
        trial_finbeats.append((count, tracklist[trial]['data'][
                                          'pt1_net_v'][
                                      start:stop].max(), period,
                               amplitude))
        count += 1

    # Remove nans...
    nans = {np.nan, float('nan')}
    trial_finbeats = [n for n in trial_finbeats if
                      not nans.intersection(n)]

    # Use for dropping altitudes
    allz = []
    for i in steady_finbeats:
        allz.append(i[0])

    if allz == []:
        min_z = 0
    else:
        min_z = min(allz)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    # fig.suptitle(trial +' '+ behavior)
    for a, b, c in steady_finbeats:
        ax1.plot(xs=[b, b], ys=[c, c], zs=[min_z, a],
                 linestyle='solid', c='black',
                 alpha=0.8, linewidth=0.5)
        ax1.scatter3D(xs=b,
                      ys=c,
                      zs=a,
                      zdir='z', s=30, marker='o',
                      c='black',
                      edgecolor='none')
    for a, b, c, d in trial_finbeats:
        ax1.plot(xs=[c, c], ys=[d, d], zs=[min_z, b],
                 linestyle='solid', c='blue',
                 alpha=0.8, linewidth=0.5)
        ax1.scatter3D(xs=c,
                      ys=d,
                      zs=b,
                      zdir='z', s=40, marker='o',
                      c='blue',
                      edgecolor='none')
        ax1.text(x=c, y=d, z=b, s=str(a), color='red', fontsize=16)

    return fig

def plot_traj_forfig(plotnum, trial, tracklist, finbeat_subset):
    """Plots a single trial's finbeats against steady finbeats 
    This function is used within a figure script to produce panels 
    for a composite figure. See "Fig6_Trial_Traj.py" for an example.
    Plots a single trial's finbeats against steady finbeats for the 
    same species at the same initial speed. 

       Args:
           plotnum (int): the number indicating the position 
                   of the panel using GridSpec conventions
           trial (string): some string identifying what's in your
                                  subset, to be used as the plot title
           tracklist (dict): the compiled position, velocity,
                             and acceleration data for all trials
                              produced by extract_data()
           finbeat_subset (): 

       Returns:
           ax1 (matplotlib Axes)
       """
    # Get info for focal trial
    behavior = tracklist[trial]['behavior']
    species = tracklist[trial]['species']
    speed = tracklist[trial]['start_spd_BLs']

    # Make the corresponding subset of steady trials
    steady_subset = make_subset(group_by1='species',
                                identifier1=species,
                                tracklist=tracklist,
                                group_by2='behavior',
                                identifier2='S',
                                group_by3='start_spd_BLs',
                                identifier3=speed)

    # Get all steady finbeats for steady subset trials
    steady_finbeats = []
    # Make (speed, period, amplitude) tuples
    for i in steady_subset:
        for j in finbeat_subset[i].index.values:
            start = finbeat_subset[i]['time'][j]
            stop = finbeat_subset[i]['endtime'][j]
            speed = tracklist[i]['data'][
                        'pt1_net_v'][start:stop].max()
            steady_finbeats.append((speed,
                                    finbeat_subset[i]['period'][j],
                                    finbeat_subset[i]['amplitude'][j]))

    # Generate tuples for the focal trial with finbeat parameters
    trial_finbeats = []
    count = 1
    for k in finbeat_subset[trial].index.values:
        start = finbeat_subset[trial]['time'][k]
        stop = finbeat_subset[trial]['endtime'][k]
        period = finbeat_subset[trial]['period'][k]
        amplitude = finbeat_subset[trial]['amplitude'][k]
        trial_finbeats.append((count, tracklist[trial]['data'][
                                          'pt1_net_v'][
                                      start:stop].max(), period,
                               amplitude))
        count += 1

    # Remove nans...
    nans = {np.nan, float('nan')}
    trial_finbeats = [n for n in trial_finbeats if
                      not nans.intersection(n)]

    # Use for dropping altitudes
    allz = []
    for i in steady_finbeats:
        allz.append(i[0])
    for i in trial_finbeats:
        allz.append(i[1])

    if allz == []:
        min_z = 0
    else:
        min_z = min(allz)

    ax1 = fig6.add_subplot(plotnum, projection='3d')
    # fig.suptitle(trial +' '+ behavior)
    for a, b, c in steady_finbeats:
        ax1.plot(xs=[b, b], ys=[c, c], zs=[min(min_z, a), max(min_z,a)],
                 linestyle='solid', c='black',
                 alpha=0.8, linewidth=0.5)
        ax1.scatter3D(xs=b,
                      ys=c,
                      zs=a,
                      zdir='z', s=30, marker='o',
                      c='black',
                      edgecolor='none')
    for a, b, c, d in trial_finbeats:
        ax1.plot(xs=[c, c], ys=[d, d], zs=[min(min_z, b),max(min_z,b)],
                 linestyle='solid', c='blue',
                 alpha=0.8, linewidth=0.5)
        ax1.scatter3D(xs=c,
                      ys=d,
                      zs=b,
                      zdir='z', s=40, marker='o',
                      c='blue',
                      edgecolor='none')
        ax1.text(x=c, y=d, z=b, s=str(a), color='red', fontsize=16)
    ax1.set_xlabel('Period(s)')
    ax1.set_ylabel('Amplitude(cm)')
    ax1.set_zlabel('Max. Inst. Speed (cm/s)')
    ax1.ticklabel_format(useOffset = False, style='sci', scilimits=(-3,2))

    return ax1

def read_data(filename):
    """Reads extract_data() and finbeat_calc() data from a pickle file 


           This function pulls tracklist, finbeats, finbeat_byP, 
           finbeat_byT from a pickle file in which they were stored, 
           without re-building them from raw file inputs. 
           Args:
               filename (str): pickle filename

           Returns:
               tracklist (dict):  stored after extract_data()
               finbeats (dict): stored after finbeat_calc()
               finbeat_byP (dict): stored after finbeat_calc() 
               finbeat_byT (dict): stored after finbeat_calc()
    """
    with open(filename, 'rb') as f:
        tracklist, finbeats, finbeat_byP, finbeat_byT = pickle.load(f)

    return tracklist, finbeats, finbeat_byP, finbeat_byT

def sin_corr(tracklist_subset, tracklist, finbeat_df):
    """Estimates best fit sine waves and returns estimates

    This estimates the frequency, amplitude, phase, and offset of a 
    sine function to best fit detrended tail tip motion from the 
    trial subset data. It then calculates the Pearson's correlation 
    coefficients between the detrended data and the sine wave. 
    Finally, it returns the parameter estimates and the correlation 
    coefficients as a dataframe.

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
            sin_corr_df (DataFrame): A Pandas DataFrame with rows= 
            trials and columns = 
                'Behavior' (str): 'S' for steady, 'A' for linear 
                                accel, 'B' for burst
                'Init_Spd' (float): initial speed in cm/s
                'Pearsons' (float): Pearson's correlation coefficient
                'Pvalue' (float): P-value for the pearson's correlation
                'Est.Freq' (float): OLS estimate for frequency
                'Est.Amplitude' (float): OLS estimate for amplitude
                'Est.Phase' (float): OLS estimate for phase
                'Est.Offset' (float): OLS estimate for offset. 


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
        period = np.mean(periods)
        guess_frequency = 1. / period
        # Estimate amplitude ( = 1/2 excursion)
        amplitudes = np.array(finbeat_df[trial]['amplitude'])
        amplitudes = abs(amplitudes)
        amplitudes = np.sort(amplitudes)
        amplitudes = amplitudes[np.logical_not(np.isnan(amplitudes))]
        amplitude = np.mean(amplitudes) / 2
        guess_amplitude = amplitude
        # Estimate y-axis and temporal shifts (arbitrary)
        guess_offset = np.mean(data)
        guess_phase = 0

        # Compose first guess sine wave (remembering sine wave freq = angular)
        data_first_guess = guess_amplitude * np.sin(
            2 * np.pi * guess_frequency * time + guess_phase) + guess_offset

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

def sort_subset(subset, tracklist):
    """Sorts a subset by behavior and speed.  

       This function sorts a subset of trials (probably created by 
       make_subset()) by behavior and initial speed. Useful for 
       organizing heatmaps. 

       Args:
           subset (list):  a subset of trials, typically 
                    generated using the convenience function 
                    make_subset()
           tracklist (dict): a tracklist produced by extract_data()

       Returns:
          (list): sorted trial names
        """

    to_sort = pd.DataFrame(data=None, index=subset, columns=[
        'behavior', 'speed'])
    for i in subset:
        to_sort['behavior'][i] = tracklist[i]['behavior']
        to_sort['speed'][i] = tracklist[i]['start_spd']

    to_sort['behavior'] = to_sort['behavior'].astype('category',
                                                     categories=[
                                                         'S', 'A',
                                                         'B'],
                                                     ordered=True)

    to_sort = to_sort.sort_values(by=['behavior', 'speed'])

    return list(to_sort.index.values)

def speed_heatmap(subset, tracklist):
    """Returns the pairwise differences in initial speed between trials 

    Used this to show relationship between cross-correlations and 
    speed. 

        Args: 
            subset (list):a subset of trials, typically 
                    generated using the convenience function 
                    make_subset()
            tracklist (dict): a tracklist produced by extract_data()
    """
    speed_diff_mat = pd.DataFrame(data=None, index=subset,
                                  columns=subset)
    for i in subset:
        speed = tracklist[i]['start_spd']
        for j in subset:
            speed2 = tracklist[j]['start_spd']

            speed_diff = np.absolute(np.subtract(speed, speed2))
            speed_diff_mat[i][j] = speed_diff

    speed_diff_mat = speed_diff_mat.apply(pd.to_numeric)

    return speed_diff_mat

def store_data(filename1, filename2):
    """Stores data in a pickle file for use later

           This function takes the items listed (intended for
           tracklist, finbeats, finbeat_byP, finbeat_byT and reg 
           models) and stores them in a pickle file so they can be 
           rapidly opened later without re-building them from raw file
           inputs. (If extract_data, finbeat_calc, and the regressions 
           are running slowly, pickle their outputs. You only need to 
           run those fns again if you add data.

           NOTE: The second file will probably be >200 MB, so you 
           cannot store it in GitHub with a basic account.

           Args:
              filename1 (str): a data filename for the pickle file
              filename2 (str): a regression filename for the pickle file

           Returns:
                Nothing
    """
    filename1 = filename1 + '.pickle'
    with open(filename1, 'wb') as f:
        pickle.dump([tracklist, finbeats, finbeat_byP, finbeat_byT], f,
                    pickle.HIGHEST_PROTOCOL)

    filename2 = filename2 + '.pickle'
    with open(filename2, 'wb') as f:
        pickle.dump(
            [T1V_model, T1V_boot_1, T2V_model, T2V_boot_1, B1V_model,
             BV_boot_1, T1A_model, T1A_boot_1, T2A_model, T2A_boot_1,
             B1A_model, BA_boot_1, T1V_model2, T1V2_boot_1, T2V_model2,
             T2V2_boot_1, B1V_model2, BV2_boot_1], f,
            pickle.HIGHEST_PROTOCOL)

