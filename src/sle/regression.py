import pandas as pd

def coef_from_sm(sm_obj):
    """Takes in a fitted statsmodels object and returns a dataframe with coefficients and confidence intervals"""

    coef_df = sm_obj.conf_int()
    coef_df["beta"] = sm_obj.params
    coef_df.drop('const', inplace = True) # drop intercept
    coef_df.columns = ["ci_lower", "ci_upper", "beta"]
    return(coef_df)