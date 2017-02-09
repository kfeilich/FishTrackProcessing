#Load all of the things
import tkinter as tk
import os
from tkinter import filedialog #For folder input popup
import pandas as pd
import numpy as np
import scipy
from scipy.signal import savgol_filter #for smoothing data

root = tk.Tk()  
root.withdraw()

#User input
folder = filedialog.askdirectory()  #Ask user for directory
framerate = float(input('Enter frame rate in frames per second:')) #Probably 500; Note assumes all videos at same framerate

         
#Initialize a list of dictionaries to contain each trial's data         
tracklist =  {}
count = 0 # Initialize the count 


for filename in os.listdir(folder): #For all of the files in the directory
    if filename.endswith("xypts.csv"): #that end with 'xypts.csv'
     
        # Extract info from filename
        filepath = folder + '/' + filename
        file_info = filename.split("_")
        fish = file_info[0]
        sequence = file_info[1]
        trial_name = fish+sequence
        
        df = pd.read_csv(filepath, sep =',')
        df = df.rename(columns = {'pt1_cam1_Y':'pt1z', 'pt1_cam2_X':'pt1x', 'pt1_cam2_Y':'pt1y', 'pt2_cam1_Y':'pt2z', 'pt2_cam2_X':'pt2x', 'pt2_cam2_Y': 'pt2y'})         
        
        # Generate time array
        df['time'] = np.linspace(0,len(df['pt1x'])*(1.0/framerate), num=len(df['pt1x']),endpoint=False)      
        df = df.set_index(['time'])
       
        # Smooth position data using savitzky golay
        df['pt1x_smth'] = scipy.signal.savgol_filter(df['pt1x'], window_length = 91, polyorder = 2)
        df['pt1y_smth'] = scipy.signal.savgol_filter(df['pt1y'], window_length = 91, polyorder = 2)
        df['pt1z_smth'] = scipy.signal.savgol_filter(df['pt1z'], window_length = 91, polyorder = 2)
        
        df['pt2x_smth'] = scipy.signal.savgol_filter(df['pt2x'], window_length = 91, polyorder = 2)
        df['pt2y_smth'] = scipy.signal.savgol_filter(df['pt2y'], window_length = 91, polyorder = 2)
        df['pt2z_smth'] = scipy.signal.savgol_filter(df['pt2z'], window_length = 91, polyorder = 2)
        
        # Calculate First Discrete Differences (Velocity)     
        cols_to_use1 = ['pt1x_smth','pt1y_smth','pt1z_smth','pt2x_smth','pt2y_smth','pt2z_smth']
        df2 = df.loc[cols_to_use].diff()        
        df2 = df2.rename(columns = {'pt1z_smth':'pt1z_v', 'pt1x_smth':'pt1x_v', 'pt1y_smth':'pt1y_v', 'pt2z_smth':'pt2z_v','pt2x_smth':'pt2x_v', 'pt2y_smth':'pt2y_v'})    
        
        # Calculate Second Discrete Differences (Acceleration)   
        cols_to_use2 = ['pt1x_v','pt1y_v','pt1z_v','pt2x_v','pt2y_v','pt2z_v']
        df3 = df2.loc[cols_to_use2].diff()
        df3 = df3.rename(columns = {'pt1z_v':'pt1z_a', 'pt1x_v':'pt1x_a', 'pt1y_v':'pt1y_a', 'pt2z_v':'pt2z_a','pt2x_v':'pt2x_a', 'pt2y_v':'pt2y_a'})

        # Merge all this shit
        df = df.merge(df2, how = 'outer', left_index=True, right_index=True)
        df = df.merge(df3, how = 'outer', left_index=True, right_index=True)        
        
        # Put all of these into the appropriate object in tracklist
        tracklist[trial_name] = {'sequence': trial_name, 'fish': fish, 'data': df}
        
        #Advance the count
        count = count + 1