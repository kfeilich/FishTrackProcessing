def boot_outputs(model, boot_list, interact=True, amplitude=True):
    model_params = ['F_test', 'Intercept', 'Period']
    if amplitude == True:
        model_params.append('Amplitude')
    if interact == True:
        model_params.append('Interact')
    boot_output = pd.DataFrame(data=None, columns=['pvalues', 'CIs95'],
                               index=model_params)

    # Return original parameter estimates
    coef_obs = model.params

    #  Find bootstrap F pvalue for whole test
    fobs = model.fvalue
    fvals = []
    for i in boot_list:
        fvals.append(i.fvalue)
    fvals_high = sum(j > fobs for j in fvals)
    boot_output['pvalues']['F_test'] = fvals_high / len(fvals)

    # Find Bootstrap p-vals for Coefficients
    # Pull all bootstrap estimates
    pvals = []
    for i in boot_list:
        pvals.append(i.pvalues)
    coefs = []
    for i in boot_list:
        coefs.append(i.params)
    tvals = []
    for i in boot_list:
        tvals.append(i.tvalues)

    # Initialize outputs
    if interact == True:
        Interact_coefs = []
        Interact_pvals = []
        Interact_tvals = []
    Period_coefs = []
    Period_pvals = []
    Period_tvals = []
    if amplitude == True:
        Amplitude_coefs = []
        Amplitude_pvals = []
        Amplitude_tvals = []
    Intercept_coefs = []
    Intercept_pvals = []
    Intercept_tvals = []

    for i in pvals:
        if interact == True:
            Interact_pvals.append(i['Period:Amplitude'])
        Period_pvals.append(i['Period'])
        if amplitude == True:
            Amplitude_pvals.append(i['Amplitude'])
        Intercept_pvals.append(i['Intercept'])

    for i in coefs:
        if interact == True:
            Interact_coefs.append(i['Period:Amplitude'])
        Period_coefs.append(i['Period'])
        if amplitude == True:
            Amplitude_coefs.append(i['Amplitude'])
        Intercept_coefs.append(i['Intercept'])

    # Confidence Intervals
    Intercept_coefs = sorted(Intercept_coefs)
    d_Intercept_coefs = Intercept_coefs - coef_obs['Intercept']
    quant_Intercept = np.percentile(a=d_Intercept_coefs, q=[5, 95])
    ci_Intercept = np.subtract(coef_obs['Intercept'], quant_Intercept)
    boot_output['CIs95']['Intercept'] = ci_Intercept

    Period_coefs = sorted(Period_coefs)
    d_Period_coefs = Period_coefs - coef_obs['Period']
    quant_Period = np.percentile(a=d_Period_coefs, q=[5, 95])
    ci_Period = np.subtract(coef_obs['Period'], quant_Period)
    boot_output['CIs95']['Period'] = ci_Period

    if amplitude == True:
        Amplitude_coefs = sorted(Amplitude_coefs)
        d_Amplitude_coefs = Amplitude_coefs - coef_obs['Amplitude']
        quant_Amplitude = np.percentile(a=d_Amplitude_coefs, q=[5, 95])
        ci_Amplitude = np.subtract(coef_obs['Amplitude'],
                                   quant_Amplitude)
        boot_output['CIs95']['Amplitude'] = ci_Amplitude

    if interact == True:
        Interact_coefs = sorted(Interact_coefs)
        d_Interact_coefs = Interact_coefs - coef_obs['Period:Amplitude']
        quant_Interact = np.percentile(a=d_Interact_coefs, q=[5, 95])
        ci_Interact = np.subtract(coef_obs['Period:Amplitude'],
                                  quant_Interact)
        boot_output['CIs95']['Interact'] = ci_Interact

    # P- Values
    p1_Intercept = sum(j > 0.0 for j in Intercept_coefs) / len(
        Intercept_coefs)
    p2_Intercept = sum(j < 0.0 for j in Intercept_coefs) / len(
        Intercept_coefs)
    boot_output['pvalues']['Intercept'] = min(p1_Intercept,
                                              p2_Intercept) * 2

    p1_Period = sum(j > 0.0 for j in Period_coefs) / len(
        Period_coefs)
    p2_Period = sum(j < 0.0 for j in Period_coefs) / len(
        Period_coefs)
    boot_output['pvalues']['Period'] = min(p1_Period, p2_Period) * 2

    if amplitude == True:
        p1_Amplitude = sum(j > 0.0 for j in Amplitude_coefs) / len(
            Amplitude_coefs)
        p2_Amplitude = sum(j < 0.0 for j in Amplitude_coefs) / len(
            Amplitude_coefs)
        boot_output['pvalues']['Amplitude'] = min(p1_Amplitude,
                                                  p2_Amplitude) * 2

    if interact == True:
        p1_Interact = sum(j > 0.0 for j in Interact_coefs) / len(
            Interact_coefs)
        p2_Interact = sum(j < 0.0 for j in Interact_coefs) / len(
            Interact_coefs)
        boot_output['pvalues']['Interact'] = min(p1_Interact,
                                                 p2_Interact) * 2

    return boot_output