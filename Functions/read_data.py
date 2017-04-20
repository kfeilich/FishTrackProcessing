import pickle

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