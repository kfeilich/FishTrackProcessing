import pickle

def store_data(filename):
    """Stores data in a pickle file for use later

           This function takes the items listed (intended for
           tracklist, finbeats, finbeat_byP, finbeat_byT) and stores
           them in a pickle file so they can be rapidly opened later
           without re-building them from raw file inputs. (If
           extract_data and finbeat_calc are running slowly, pickle
           their outputs. You only need to run those fns again if
           you add data.
           Args:
               items (list): a list of strings with the dictionaries
                            you would like to store. Generally [
                            tracklist, finbeats, finbeat_byP,
                            finbeat_byT]

           Returns:
                Nothing
    """
    filename = filename + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump([tracklist, finbeats,finbeat_byP, finbeat_byT] , f,
                     pickle.HIGHEST_PROTOCOL)
