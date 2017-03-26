import pandas as pd

def sort_subset(subset, tracklist):
    to_sort = pd.DataFrame(data=None, index=subset, columns=[
        'behavior', 'speed'])
    for i in subset:
        to_sort['behavior'][i] = tracklist[i]['behavior']
        to_sort['speed'][i] = tracklist[i]['start_spd']

    to_sort['behavior'] = to_sort['behavior'].astype('category')
    to_sort['behavior'].cat.reorder_categories(['S', 'A', 'B'],
                                               ordered=True)

    to_sort = to_sort.sort_values(by=['behavior', 'speed'],
                                  ascending=True)

    return list(to_sort.index.values)

