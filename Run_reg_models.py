import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
from statsmodels.graphics.regressionplots import plot_leverage_resid2


# Prep
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

# ByP Models: Full Factorial
T1V_model = smf.ols('Max_Spd ~ Period * Amplitude', data = trout1_params_byP).fit()
T1V_boot_1 = mult_reg_boot(formula='Max_Spd ~ Period * Amplitude',y='Max_Spd', df=trout1_params_byP,frac_sub = 1.0, reps = 2000)
T1A_model = smf.ols('Max_Acc ~ Period * Amplitude', data = trout1_params_byP).fit()
T1A_boot_1= mult_reg_boot(formula='Max_Acc ~ Period * Amplitude',y='Max_Acc', df=trout1_params_byP,frac_sub = 1.0, reps = 2000)

T2V_model = smf.ols('Max_Spd ~ Period * Amplitude', data = trout2_params_byP).fit()
T2V_boot_1 = mult_reg_boot(formula='Max_Spd ~ Period * Amplitude',y='Max_Spd', df=trout2_params_byP,frac_sub = 1.0, reps = 2000)
T2A_model = smf.ols('Max_Acc ~ Period * Amplitude', data = trout2_params_byP).fit()
T2A_boot_1 = mult_reg_boot(formula='Max_Acc ~ Period * Amplitude',y='Max_Acc', df=trout2_params_byP,frac_sub = 1.0, reps = 2000)

B1V_model = smf.ols('Max_Spd ~ Period * Amplitude', data = bass_params_byP).fit()
BV_boot_1 = mult_reg_boot(formula='Max_Spd ~ Period * Amplitude',y='Max_Spd', df=bass_params_byP,frac_sub = 1.0, reps = 2000)
B1A_model = smf.ols('Max_Acc ~ Period * Amplitude', data = bass_params_byP).fit()
BA_boot_1= mult_reg_boot(formula='Max_Acc ~ Period * Amplitude',y='Max_Acc', df=bass_params_byP,frac_sub = 1.0, reps = 2000)

T1V_boot_outputs = boot_outputs(T1V_model, T1V_boot_1)
print('Trout1_MaxV')
print(T1V_boot_outputs)

T2V_boot_outputs = boot_outputs(T2V_model, T2V_boot_1)
print('Trout2_MaxV')
print(T2V_boot_outputs)

BV_boot_outputs = boot_outputs(B1V_model, BV_boot_1)
print('Bass_MaxV')
print(BV_boot_outputs)

T1A_boot_outputs = boot_outputs(T1A_model, T1A_boot_1)
print('Trout1_MaxA')
print(T1A_boot_outputs)

T2A_boot_outputs = boot_outputs(T2A_model, T2A_boot_1)
print('Trout2_MaxA')
print(T2A_boot_outputs)

BA_boot_outputs = boot_outputs(B1A_model, BA_boot_1)
print('Bass_MaxA')
print(BA_boot_outputs)

# By P No interactions
T1V_model2 = smf.ols('Max_Spd ~ Period + Amplitude', data = trout1_params_byP).fit()
T1V2_boot_1 = mult_reg_boot(formula='Max_Spd ~ Period + Amplitude',y='Max_Spd', df=trout1_params_byP,frac_sub = 1.0, reps = 2000)
print(boot_output(T1V_model2, T1V2_boot_1, interact=False,
                amplitude=True))

#T2V_model2 = smf.ols('Max_Spd ~ Period + Amplitude', data = trout2_params_byP).fit()
#T2V2_boot_1 = mult_reg_boot(formula='Max_Spd ~ Period + Amplitude',y='Max_Spd', df=trout2_params_byP,frac_sub = 1.0, reps = 1000)
#print(boot_pvals_np(T2V_model2, T2V2_boot_1, interact=False, amplitude=True))


B1V_model2 = smf.ols('Max_Spd ~ Period + Amplitude', data = bass_params_byP).fit()
BV2_boot_1 = mult_reg_boot(formula='Max_Spd ~ Period + Amplitude',y='Max_Spd', df=bass_params_byP,frac_sub = 1.0, reps = 2000)
print(boot_output(B1V_model2, BV2_boot_1, interact=False,
                 amplitude=True))


"""

filename = 'mult_reg_models.pickle'
with open(filename, 'wb') as f:
        pickle.dump([T1V_model, T1V_boot_1, T2V_model, T2V_boot_1, B1V_model, BV_boot_1, T1A_model, T1A_boot_1, T2A_model, T2A_boot_1, B1A_model, BA_boot_1, T1V_model2, T1V2_boot_1, B1V_model2, BV2_boot_1] , f,
                     pickle.HIGHEST_PROTOCOL)
                     """