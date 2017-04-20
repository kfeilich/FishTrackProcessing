import pandas as pd
import numpy as np

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