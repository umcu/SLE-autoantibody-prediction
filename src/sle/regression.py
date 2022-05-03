"""Extract coefficients from regression models into plots / dataframes.

    See the main() function and/or run this module as a script for a typical usage example.
"""

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
    """Combine a fitted statsmodels regression model with a regularized (e.g. cross-validated lasso) regression model from sklearn,
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

    tbl_coef = (pd.Series(sk_results.intercept_, index = ['const'])
                .append(pd.Series(sk_results.coef_.squeeze(), index = varnames))
                .to_frame(name='coef_regularized')
                .join(sm_df))
    tbl_coef = tbl_coef.round(3)
    tbl_coef["95%CI"] = tbl_coef["[0.025"].astype(str) + ", " + tbl_coef["0.975]"].astype(str)
    tbl_coef.drop(columns=["z", "P>|z|", "[0.025", "0.975]"], inplace=True)
    return tbl_coef


def main():

    import statsmodels.api as sm
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    dat = load_breast_cancer(as_frame=True)
    df = dat.frame
    X = df.drop(columns='target')
    y = df['target']

    sm_results = sm.Logit(y,sm.add_constant(X)).fit(method='bfgs')

    # Extract coefficients and confidence intervals into data frame
    coef_from_sm(sm_results)

    # Plot coefficients and confidence intervals
    coef_plot(coef_from_sm(sm_results))

    # Combine coefficients from statsmodels and sklearn (regularized model)
    # into one data frame
    model = LogisticRegression(solver='liblinear')
    sk_results = model.fit(X,y)

    make_coef_tbl(sm_results, sk_results, X.columns)


if __name__ == '__main__':
    main()
