import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


def mult_reg_boot(formula, y, df, frac_sub, reps):
    """Runs bootstrap reps of multiple linear regression to est outputs

           Args:
                formula (str): a list of strings with the trial
                                   names of the desired trials from tracklist.
                                    Note: The list (even of a single
                                    element) must be contained in square
                                    brackets.
                                    Also note: Alternatively, to iterate
                                    over all trials, set this to
                                    tracklist.keys()
               y (pd.DataFrame series): response variable
               df (Pandas DataFrame):
               size_sub (int): size of subset
               reps (int): number of bootstrap replicates
               
           Returns:
    """
    # Make a list to hold model objects
    # Method 1: regression case resampling
    # Method 2: Resampling residuals

    boot_outputs_method1 = []
    boot_outputs_method2 = []

    # Method 1
    # Iterate model runs over the reps
    for rep in np.arange(1,reps):
        subset = df.sample(frac=frac_sub, replace=True)
        reg_model1 = smf.ols(formula=formula, data = subset).fit()
        boot_outputs_method1.append(reg_model1)

    # Method 2
    reg_model2 = smf.ols(formula=formula, data=df).fit()
    predicted = reg_model2.fitted
    residuals = reg_model2.resid
    stu_resid = reg_model2.wresid
    pearson_resid = reg_model2.resid_pearson

    for rep in np.arange(1,reps):
        df_copy = df.copy()
        # randomly resample residuals and add these random resids to y
        random_resid = np.random.choice(residuals, size=len(
            residuals), replace=True)
        df_copy[y] = df_copy[y] + random_resid
        # refit model using fake y and store output
        reg_model2_rep = smf.ols(formula=formula, data=df_copy).fit()
        boot_outputs_method2.append(reg_model2_rep)

    return boot_outputs_method1, boot_outputs_method2
