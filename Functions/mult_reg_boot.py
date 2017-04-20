import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


def mult_reg_boot(formula, y, df, reps=2000):
    """Runs bootstrap reps of multiple linear regression to est outputs

           Args:
                formula (str): string identifying the formula as 
                        input to stats models, typically of the form
                         'Y ~ X1 * X2' , or something like that. 
                         See the Jupyter notebook for examples. 
               y (string): string identifying the Pandas Series within 
                        df that contains the response variable
               df (Pandas DataFrame): one of the dataframes prepared 
                        using script finbeat_params_prep.py
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
