from __future__ import division, print_function
# Load all of the things
import tkinter as tk
import os
import pandas as pd
import numpy as np
import scipy
import matplotlib.cm as cm
import matplotlib
from scipy.signal import savgol_filter  # for smoothing data
from mpl_toolkits.mplot3d import Axes3D
from tkinter import filedialog  # For folder input popup
matplotlib.rc('axes.formatter', useoffset=False)

root = tk.Tk()
root.withdraw()

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