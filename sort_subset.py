import pandas as pd

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
                                                     categories = [
                                                         'S', 'A',
                                                         'B'],
                                                     ordered = True)

    to_sort = to_sort.sort_values(by=['behavior', 'speed'])

    return list(to_sort.index.values)

