import pandas as pd
import numpy as np

n_byP = 0
n_byT = 0

for trial in finbeat_byP:
    for fb in finbeat_byP[trial].index:
        n_byP += 1

for trial in finbeat_byT:
    for fb in finbeat_byT[trial].index:
        n_byT += 1

finbeat_params_byP = pd.DataFrame(data=None, index=np.arange(0, n_byP),
                                  columns=["Index", "Species", "Fish",
                                           "Trial",
                                           "FB_num", "Period",
                                           "Amplitude",
                                           "Init_Spd", "Max_Spd",
                                           "Max_Acc"])

finbeat_params_byT = pd.DataFrame(data=None, index=np.arange(0, n_byT),
                                  columns=["Index", "Species", "Fish",
                                           "Trial",
                                           "FB_num", "Period",
                                           "Amplitude",
                                           "Init_Spd", "Max_Spd",
                                           "Max_Acc"])
count = 0

for trial in finbeat_byP:
    for fb in finbeat_byP[trial].index:
        finbeat_params_byP['Index'][count] = count
        finbeat_params_byP['Species'][count] = tracklist[trial][
            'species']
        finbeat_params_byP['Fish'][count] = tracklist[trial]['fish']
        finbeat_params_byP['Trial'][count] = str(trial)
        finbeat_params_byP['FB_num'][count] = fb
        finbeat_params_byP['Period'][count] = \
        finbeat_byP[trial]['period'][fb]
        finbeat_params_byP['Amplitude'][count] = \
        finbeat_byP[trial]['amplitude'][fb]
        finbeat_params_byP['Init_Spd'][count] = tracklist[trial][
            'start_spd']

        start = finbeat_byP[trial]['time'][fb]
        end = finbeat_byP[trial]['endtime'][fb]

        finbeat_params_byP['Max_Spd'][count] = tracklist[trial]['data'][
                                                   'pt1_net_v'][
                                               start:end].max()
        finbeat_params_byP['Max_Acc'][count] = tracklist[trial]['data'][
                                                   'pt1_net_a'][
                                               start:end].max()
        count += 1

count = 0
for trial in finbeat_byT:
    for fb in finbeat_byT[trial].index:
        finbeat_params_byT['Index'][count] = count
        finbeat_params_byT['Species'][count] = tracklist[trial][
            'species']
        finbeat_params_byT['Fish'][count] = tracklist[trial]['fish']
        finbeat_params_byT['Trial'][count] = str(trial)
        finbeat_params_byT['FB_num'][count] = fb
        finbeat_params_byT['Period'][count] = \
        finbeat_byT[trial]['period'][fb]
        finbeat_params_byT['Amplitude'][count] = \
        finbeat_byT[trial]['amplitude'][fb]
        finbeat_params_byT['Init_Spd'][count] = tracklist[trial][
            'start_spd']

        start = finbeat_byT[trial]['time'][fb]
        end = finbeat_byT[trial]['endtime'][fb]

        finbeat_params_byT['Max_Spd'][count] = tracklist[trial]['data'][
                                                   'pt1_net_v'][
                                               start:end].max()
        finbeat_params_byT['Max_Acc'][count] = tracklist[trial]['data'][
                                                   'pt1_net_a'][
                                               start:end].max()
        count += 1

finbeat_params_byP[['Period', 'Amplitude', 'Max_Spd', 'Max_Acc']] = finbeat_params_byP[['Period', 'Amplitude', 'Max_Spd', 'Max_Acc']].apply(pd.to_numeric, errors = 'coerce')
trout1_params_byP = finbeat_params_byP.loc[finbeat_params_byP['Fish']=='BTrout1']
trout1_params_byP = trout1_params_byP.dropna()

trout2_params_byP = finbeat_params_byP.loc[finbeat_params_byP['Fish']=='BTrout2']
trout2_params_byP = trout2_params_byP.dropna()

bass_params_byP = finbeat_params_byP.loc[finbeat_params_byP['Species']=='Bass']
bass_params_byP = bass_params_byP.dropna()

finbeat_params_byT[['Period', 'Amplitude', 'Max_Spd', 'Max_Acc']] = finbeat_params_byT[['Period', 'Amplitude', 'Max_Spd', 'Max_Acc']].apply(pd.to_numeric, errors = 'coerce')

trout1_params_byT = finbeat_params_byT.loc[finbeat_params_byT['Fish']=='BTrout1']
trout1_params_byT = trout1_params_byT.dropna()
trout2_params_byT = finbeat_params_byT.loc[finbeat_params_byT['Fish']=='BTrout2']
trout2_params_byT = trout2_params_byT.dropna()
bass_params_byT = finbeat_params_byT.loc[finbeat_params_byT['Species']=='Bass']
bass_params_byT = bass_params_byT.dropna()