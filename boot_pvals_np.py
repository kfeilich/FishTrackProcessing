def boot_pvals_np(model, boot_list):
    model_params = ['F_test', 'Intercept', 'Period', 'Amplitude',
                    'Interact']
    boot_pvalues=pd.Series(data=None, index=model_params)

    #  Find bootstrap F pvalue for whole test
    fobs = model.fvalue
    fvals = []
    for i in boot_list:
        fvals.append(i.fvalue)
    fvals_high = sum(j > fobs for j in fvals)
    boot_pvalues['F_test'] = fvals_high / len(fvals)

    # Find Bootstrap p-vals for Coefficients
    # Pull all bootstrap estimates
    pvals = []
    for i in boot_list:
        pvals.append(i.pvalues)
    coefs = []
    for i in boot_list:
        coefs.append(i.params)

    # Initialize outputs
    Interact_coefs = []
    Interact_pvals = []
    Period_coefs = []
    Period_pvals = []
    Amplitude_coefs = []
    Amplitude_pvals = []
    Intercept_coefs = []
    Intercept_pvals = []

    for i in pvals:
        Interact_pvals.append(i['Period:Amplitude'])
        Period_pvals.append(i['Period'])
        Amplitude_pvals.append(i['Amplitude'])
        Intercept_pvals.append(i['Intercept'])

    for i in coefs:
        Interact_coefs.append(i['Period:Amplitude'])
        Period_coefs.append(i['Period'])
        Amplitude_coefs.append(i['Amplitude'])
        Intercept_coefs.append(i['Intercept'])

    p1_Intercept = sum(j > 0.0 for j in Intercept_coefs) / len(
        Intercept_coefs)
    p2_Intercept = sum(j < 0.0 for j in Intercept_coefs) / len(
        Intercept_coefs)
    boot_pvalues['Intercept'] = min(p1_Intercept, p2_Intercept) * 2

    p1_Period = sum(j > 0.0 for j in Period_coefs) / len(
        Period_coefs)
    p2_Period = sum(j < 0.0 for j in Period_coefs) / len(
        Period_coefs)
    boot_pvalues['Period'] = min(p1_Period, p2_Period) * 2

    p1_Amplitude = sum(j > 0.0 for j in Amplitude_coefs) / len(
        Amplitude_coefs)
    p2_Amplitude = sum(j < 0.0 for j in Amplitude_coefs) / len(
        Amplitude_coefs)
    boot_pvalues['Amplitude'] = min(p1_Amplitude, p2_Amplitude) * 2

    p1_Interact = sum(j > 0.0 for j in Interact_coefs) / len(
        Interact_coefs)
    p2_Interact = sum(j < 0.0 for j in Interact_coefs) / len(
        Interact_coefs)
    boot_pvalues['Interact'] = min(p1_Interact, p2_Interact) * 2

    return boot_pvalues