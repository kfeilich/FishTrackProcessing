import pickle

def read_data(filename):
    """Stores data in a pickle file for use later

           This function takes the items listed (intended for
           tracklist, finbeats, finbeat_byP, finbeat_byT) and stores
           them in a pickle file so they can be rapidly opened later
           without re-building them from raw file inputs. (If
           extract_data and finbeat_calc are running slowly, pickle
           their outputs. You only need to run those fns again if
           you add data.
           Args:
               filename (str): pickle filename

           Returns:
                Nothing
    """
    with open(filename, 'rb') as f:
        tracklist, finbeats, finbeat_byP, finbeat_byT = pickle.load(f)

    return tracklist, finbeats, finbeat_byP, finbeat_byT