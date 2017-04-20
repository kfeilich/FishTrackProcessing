import pickle

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
        pickle.dump([tracklist, finbeats,finbeat_byP, finbeat_byT] , f,
                     pickle.HIGHEST_PROTOCOL)

    filename2 = filename2 + '.pickle'
    with open(filename2, 'wb') as f:
        pickle.dump(
            [T1V_model, T1V_boot_1, T2V_model, T2V_boot_1, B1V_model,
             BV_boot_1, T1A_model, T1A_boot_1, T2A_model, T2A_boot_1,
             B1A_model, BA_boot_1, T1V_model2, T1V2_boot_1, T2V_model2,
             T2V2_boot_1, B1V_model2, BV2_boot_1], f,
            pickle.HIGHEST_PROTOCOL)