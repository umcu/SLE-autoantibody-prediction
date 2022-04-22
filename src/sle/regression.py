from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def coef_from_sm(sm_obj):
    """Takes in a fitted statsmodels object and returns a dataframe with
    coefficients and confidence intervals.
    """

    coef_df = sm_obj.conf_int()
    coef_df["beta"] = sm_obj.params
    coef_df.drop('const', inplace = True) # drop intercept
    coef_df.columns = ["ci_lower", "ci_upper", "beta"]
    return coef_df


def coef_plot(coef_df, OR: bool = False):
    """Given a dataframe of coefficients (from coef_from_sm()),
    plot the point estimates and confidence intervals.

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


def make_coef_tbl(sm_results, sk_results, varnames: List[str]) -> pd.DataFrame:
    """Combine a fitted statsmodels regression model with a regularized (cross-validated lasso) regression model from sklearn, 
    and create one table with the coefficients, as well as standard errors and confidence intervals (only for non-regularized model)

    Args:
        sm_results: fitted statsmodels object
        sk_results: fitted sklearn object
        varnames: list of feature names
    
    Returns:
        dataframe with table
    """
    results_as_html = sm_results.summary().tables[1].as_html()
    sm_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    
    tbl_coef = (pd.Series(sk_results.best_estimator_.named_steps.clf.intercept_, index = ['const'])
                .append(pd.Series(sk_results.best_estimator_.named_steps.clf.coef_.squeeze(), index = varnames))
                .to_frame(name='coef_lasso')
                .join(sm_df))
    tbl_coef = tbl_coef.round(3)
    tbl_coef["95%CI"] = tbl_coef["[0.025"].astype(str) + ", " + tbl_coef["0.975]"].astype(str)
    tbl_coef.drop(columns=["z", "P>|z|", "[0.025", "0.975]"], inplace=True)
    return tbl_coef