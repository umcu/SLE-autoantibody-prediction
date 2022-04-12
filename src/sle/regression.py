import numpy as np
import matplotlib.pyplot as plt

def coef_from_sm(sm_obj):
    """Takes in a fitted statsmodels object and returns a dataframe with coefficients and confidence intervals"""

    coef_df = sm_obj.conf_int()
    coef_df["beta"] = sm_obj.params
    coef_df.drop('const', inplace = True) # drop intercept
    coef_df.columns = ["ci_lower", "ci_upper", "beta"]
    return(coef_df)


def coef_plot(coef_df, OR: bool = False):
    """Given a dataframe of coefficients (from coef_from_sm()), plot the point estimates and confidence intervals.
    
    Args:
        coef_df: dataframe from coef_from_sm()
        OR: boolean whether you're passing betas (default; False) or odds ratios (True)
    
    """

    ref_line=0
    xlab=r'$\beta$'
    if OR:
        ref_line=1
        xlab='OR'
    coef_df = coef_df.sort_values(by='beta')
    plt.figure(figsize=(7,10))
    plt.errorbar(y=coef_df.index, x=coef_df.beta, 
             xerr= np.array([coef_df.beta-coef_df.ci_lower, coef_df.ci_upper-coef_df.beta]),
             fmt='ok',
             ecolor='gray')
    plt.axvline(x=ref_line, color='k', linestyle='--')
    plt.xlabel(xlab)