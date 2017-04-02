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
    # Method 1: Resampling residuals

    boot_outputs_method1 = []

    # Method 1
    reg_model1 = smf.ols(formula=formula, data=df).fit()
    predicted = reg_model1.fittedvalues
    residuals = reg_model1.resid
    stu_resid = reg_model1.wresid
    pearson_resid = reg_model1.resid_pearson

    for rep in np.arange(1, reps):
        df_copy = df.copy()
        # randomly resample residuals and add these random resids to y
        random_resid = np.random.choice(residuals, size=len(
            residuals), replace=True)
        df_copy[y] = df_copy[y] + random_resid
        # refit model using fake y and store output
        reg_model1_rep = smf.ols(formula=formula, data=df_copy).fit()
        boot_outputs_method1.append(reg_model1_rep)

    return boot_outputs_method1
