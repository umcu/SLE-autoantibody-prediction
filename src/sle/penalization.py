"""Functions to assist with fitting, evaluation, and plotting regularized regression models.

    See the main() function and/or run this module as a script for a typical usage example.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.base import clone


def regularization_range(X: pd.DataFrame, y: pd.Series, scaler=None, alpha: float = 1, epsilon: float = 0.001, as_glmnet: bool = False
) -> Tuple[float, float]:
    """ Returns minimum and maximum values for lamdbda (regularization strength) to evaluate in regularized regression,
    as is computed by the R package glmnet
    From: https://stats.stackexchange.com/a/270786/95723

    N.B. This implementation gives the same result as l1_min_c from sklearn.svm. However, here we output lambda (as in glmnet), whereas
    sklearn uses C (the inverse of lambda) for regularization strength. So don't forget to convert if you have to: C = 1/lambda.

    Args:
        X: a pandas dataframe with features
        y: a pandas series with class labels
        scaler: an sklearn.preprocessing method (e.g. StandardScaler())
        alpha: type of elastic net penalty (alpha = 1 for pure L1/LASSO, alpha = 0 for pure L2/Ridge).
               N.B. For Ridge, don't use alpha = 0 exactly, as then lamda_max will be infinite. glmnet default for Rdige is alpha = 0.001
        epsilon: the range between lambda_min and lambda_max

    Returns:
        lambda_min: lowest regularization, defined as epsilon*lambda_max
        lambda_max: highest regularization. For LASSO, all coefficients will be exactly 0 for a model with lambda_max
    """
    if scaler:
        sx = scaler.fit_transform(X) # normalize feature matrix
    else:
        sx = X.to_numpy()

    norm_term = 1
    if as_glmnet:
        norm_term = len(y) # add this term to get same lambda_max as glmnet

    lambda_max = np.max(np.abs((y-np.mean(y)*(1-np.mean(y))).T @ sx)) /alpha /norm_term
    lambda_min = epsilon * lambda_max

    return lambda_min, lambda_max


def choose_C(results, return_index: bool = True):
    """Choose a final value for the regularization parameter based on the "1 SE rule",
    i.e. not the value with the highest cross-validated performance, but the value that's
    one standard deviation higher. This is also implemented in the R package glmnet, and serves
    to prevent under-regularization due to uncertainty about what's the best performing model.

    adapted from https://github.com/scikit-learn-contrib/lightning/issues/84#issuecomment-537064107

    Args:
        results: the cv_results_ attribute of the GridSearchCV object
        return_index: if False, returns C itself
                      if True (default), returns the index of the best-performing C-value in the object.
                      This way, this function can simply be passed to the refit parameter in GridSeachCV

    Returns:
        best_C_index (when return_index = True), or best_C (when return_index = False)

    Recommended usage (when return_index = True):

    GridSearchCV(estimator, param_grid, refit = choose_C)
    """

    K = len([x for x in list(results.keys()) if x.startswith('split')]) # number of splits in CV
    C_range = results['param_clf__C'].data # values for regularization parameter C we tried

    mean_per_C = results['mean_test_score'] # score per C
    sem_per_C = results['std_test_score'] / np.sqrt(K) # standard error of the mean

    best_score = np.max(mean_per_C) # best score value
    sem = sem_per_C[np.argmax(mean_per_C)] # SEM of this score

    C_candidates = mean_per_C >= best_score - sem # boolean for all values of C that exceed SEM of the best score
    best_C_index = np.where(C_candidates)[0][-1] # find last index in this list (strongest regularization)
    best_C = C_range[best_C_index] # get this value for C

    if return_index:
        return best_C_index
    else:
        return best_C


def regularization_path(reg_param: np.ndarray, clf, X: pd.DataFrame, y: pd.Series
) -> Tuple[List[np.ndarray], List[int]]:
    """Compute the regularization path (model coefficients for different regularization values)

    from https://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py

    Args:
        reg_param: a numpy array of regularization_values (C) to evaluate
        clf: an sklearn model (e.g. LogisticRegression) with C as a hyperparameter
        X: a dataframe with features
        y: a series with class labels

    Returns:
        coefs: a list of numpy arrays with model coefficients, one array for each C value
        nnz_coefs: a list of ints with the number of non-zero model coefficients, one int for each C value

    """
    # clone the model to set warm_start to true, which saves time here for repeated fitting
    # but we don't want when fitting the model normally
    clf_warm = clone(clf)
    clf_warm.set_params(warm_start=True)
    coefs = []
    nnz_coefs = []
    for r in reg_param:
        clf_warm.set_params(C=r)
        clf_warm.fit(X,y)
        coefs.append(clf_warm.coef_)
        nnz_coefs.append(np.sum(np.abs(clf_warm.coef_) > 0))
    return coefs, nnz_coefs


def plot_regularization_path(reg_param: np.ndarray, coefs: List[np.ndarray], nnz_coefs: List[str], results, num_ticks: int = 10):
    """
    Plot the regularization path. Produces two subplots:
    1. lambda on the x-axis, model coefficients on the y-axis
    2. lambda on the x-axis (plus 2nd x-axis with number of non-zero coefficients), model performance on the y-axis

    The first dotted line shows lambda with best performance;
    the 2nd dotted line shows lambda at 1 standard error below best performance

    Args:
        reg_param: same as in regularization_path()
        coefs: output of regularization_path()
        nnz_coefs: output of regularization_path()
        results: cv_results_ attribute of the output returned by GridSearchCV
        num_ticks: number of ticks for nnz_coefs values to put on second axis

    Returns:
        ax1: handle to first subplot
        ax2: handle to 2nd subplot
        ax22: handle to 2nd x-axis in 2nd subplot
    """
    K = len([x for x in list(results.keys()) if x.startswith('split')]) # number of splits in CV
    mean_score = results['mean_test_score']
    SEM_score = results['std_test_score'] / np.sqrt(K)

    plt.figure(figsize=(7,10))
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(reg_param, np.array(coefs).squeeze())
    ax1.set_xscale('log')
    plt.ylabel('Betas')

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(reg_param, mean_score)
    ax2.fill_between(reg_param, mean_score - SEM_score, mean_score + SEM_score, color='grey', alpha=.2, label=r'$\pm$ 1 SEM')
    plt.axvline(x=reg_param[np.argmax(mean_score)], color='k', linestyle='--') # C with best score
    plt.axvline(x=reg_param[choose_C(results)], color='k', linestyle='--') # highest value for C within 1 SE of best score
    ax2.set_xscale('log')
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('ROC AUC')
    ax2.legend()

    ax22 = ax2.twiny()
    ax22.set_xscale('log')
    ax22.set_xlim(ax2.get_xlim())
    ax22.set_xticks(reg_param[0::num_ticks+1]) # display 10 ticks by default
    ax22.set_xticklabels(nnz_coefs[0::num_ticks+1])
    ax22.set_xlabel('n features retained')

    plt.tight_layout()

    return ax1, ax2, ax22


def coef_plots_regularized(coefs: List[np.ndarray], nnz_coefs: List[int], scores, varnames: List[str], num_subplot_cols: int = 5):
    """For a LASSO model, draw a grid of coeffcient plots: one for each value of lambda that led to a different set of coefficients.
    Starts at the model with the best performance, and continues until only a single feature is left.

    Args:
        coefs: output of regularization_path()
        nnz_coefs: output of regularization_path()
        scores: cv_results_["mean_test_score"] attribute of the GridSearchCV output. N.B. Assumes this is ROC AUC
        varnames: list of feature names
    """
    best_num_features = nnz_coefs[np.argmax(scores)] # number of features of model with best ROC AUC
    best_idx = np.argmax(np.unique(nnz_coefs) == best_num_features) # find index of this model in vector with feature counts for each model

    features = np.unique(nnz_coefs)[best_idx:0:-1] # for this number of features, and each unique number below it
    num_rows = int(np.ceil(len(features)/num_subplot_cols))

    # subplot code from https://stackoverflow.com/a/31575923
    gs = gridspec.GridSpec(num_rows, num_subplot_cols)
    fig = plt.figure(figsize=(5*num_rows, 3*num_subplot_cols))
    for n, num_features in enumerate(features):
        idx = np.argmax(nnz_coefs == num_features) # find first model that has this number of features
        df_coefs = pd.Series(coefs[idx].squeeze(), index=varnames) # get coefficients, add feature names
        ax = fig.add_subplot(gs[n])
        df_coefs[df_coefs != 0].sort_values().plot.barh(ax=ax) # get non-zero coefficients, sort descending
        ax.set_title(f'Num features: {num_features}; AUC: {scores[idx]:.4f}')

    fig.tight_layout()

def main():

    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dat = load_breast_cancer(as_frame=True)
    df = dat.frame
    X = df.drop(columns='target')
    y = df['target']

    cv = StratifiedKFold(shuffle=True, random_state=40)
    scaler = StandardScaler()
    lr = LogisticRegression(solver='liblinear', penalty='l1')

    X_sc = scaler.fit_transform(X)

    # Pick optimal hyperparameter range
    lambda_min, lambda_max = regularization_range(X,y,scaler)

    Cs = np.logspace(np.log10(1/lambda_min), np.log10(1/lambda_max), 100)
    pipe = Pipeline([
        ('scaler', scaler),
        ('clf', lr)
    ])
    params = [{
    "clf__C": Cs
    }]

    lr_l1 = GridSearchCV(pipe, params, cv = cv, scoring = 'roc_auc', refit=choose_C)

    lr_l1.fit(X,y)

    # Compute and plot regularization path
    coefs, nnz_coefs = regularization_path(Cs, lr, X_sc, y)
    plot_regularization_path(1/Cs, coefs, nnz_coefs, lr_l1.cv_results_)

    # Plot coefficients at different regularization strengths
    coef_plots_regularized(coefs, nnz_coefs, lr_l1.cv_results_["mean_test_score"], X.columns, num_subplot_cols=3)

if __name__ == '__main__':
    main()
