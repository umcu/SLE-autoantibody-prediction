""""General routines to prep data for predictive modeling, and to evaluate resulting models.

    See the main() function and/or run this module as a script for a typical usage example.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.metrics import  auc, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, plot_roc_curve


def generate_data(which: str) -> pd.DataFrame:
    """Generate synthetic data to able to run the code in this project.

    This creates a dataframe with the same shape, column names, and types of features as the real data.
    However, the values themselves are generated from simple distributions: correlations between "features"
    are not preserved, nor is which features are most predictive, and their values don't necessarily make sense.

    Args:
        which: str (either 'imid', or 'rest'), which determines which of two dataframes to generate:
            'imid' generates a dataframe with classes "SLE", "BBD", "IMID", and "nonIMID" (similar to imid.feather)
            'rest' generate a datafrae with classes "rest_large", "rest", "LLD" "preSLE" (similar to rest.feather)

    Returns:
        a dataframe that satisfies the properties described above
    """

    if which not in ['imid', 'rest']:
        raise ValueError(f"argument must be one of ['imid','rest']; input was '{which}' instead")

    total_n = {'imid': 1408,
               'rest': 922}

    num_feature_names = ['Actinin', 'ASCA', 'Beta2GP1', 'C1q', 'C3b', 'Cardiolipin', 'CCP1arg',
       'CCP1cit', 'CENP', 'CMV', 'CollagenII', 'CpGmot', 'CRP1', 'DFS70',
       'dsDNA2', 'Enolasearg', 'Enolasecit', 'EphB2', 'FcER', 'Fibrillarin',
       'Ficolin', 'GAPDH', 'GBM', 'H2Bp', 'H2Bpac', 'H4p', 'H4pac', 'Histones',
       'IFNLambda', 'IFNOmega', 'Jo1', 'Ku', 'LaSSB', 'MBL2', 'Mi2',
       'Nucleosome', 'PCNA', 'Pentraxin3', 'PmScl100', 'RA33', 'RipP0',
       'RipP0peptide', 'RipP1', 'RipP2', 'RNAPolIII', 'RNP70', 'RNPA', 'RNPC',
       'Ro52', 'Ro60', 'RPP25ThTo', 'Scl70', 'SmBB', 'SMP', 'TIF1gamma', 'TPO',
       'tTG', 'dsDNA1']

    class_counts = {'imid': {
                         'SLE':        483,
                         'BBD':        361,
                         'IMID':       346,
                         'nonIMID':    218},
                    'rest': {
                         'rest_large': 462,
                         'rest':       415,
                         'LLD':         28,
                         'preSLE':      17}
                  }

    n_symps = class_counts['imid']['SLE'] + class_counts['imid']['IMID']
    # approximate proportion of symptom occurences in real data SLE/IMID patients
    symptom_props = {'Arthritis': 180/n_symps,
                     'Pleurisy': 86/n_symps,
                     'Pericarditis': 85/n_symps,
                     'Nefritis': 172/n_symps}

    # generate numerical features
    X,y = make_classification(n_samples=total_n[which], n_features = len(num_feature_names), n_informative=len(class_counts[which])*2,
                              n_classes=len(class_counts[which]), weights = [c/total_n[which] for c in class_counts[which].values()],
                              random_state=40)
    df = pd.DataFrame(X, columns = num_feature_names)
    df = df.abs() # force positive (as fluorescence intensity can't be negative)
    df['Class'] = pd.Series(y).replace(list(range(len(class_counts[which]))), class_counts[which].keys()) # replace 0-3 with class labels

    # add binary features
    for s, c in symptom_props.items():
        df[s] = np.nan
        # assign symptoms (1 vs. 0) to SLE/IMID in proportion to real data
        if which == 'imid':
            df.loc[df.Class.isin(['SLE', 'IMID']), s] = np.random.binomial(1, c, df.Class.isin(['SLE', 'IMID']).sum())
            df.loc[df.Class == 'nonIMID', s] = 0 # nonIMIDs (almost) never have these symptoms
        elif which == 'rest':
            df.loc[df.Class.isin(['LLD', 'preSLE']), s] = np.random.binomial(1, c, df.Class.isin(['LLD', 'preSLE']).sum())
    return df


def prep_data(df: pd.DataFrame, target_class: str, control_class: str, drop_cols: list = []
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate X and y from data frame for machine learning.

    Args:
        df: A pandas dataframe, with a "Class" column
        target_class: a string specifying the name of the target class in the "Class" column
        control_class: a string specifying the name of the negative class in the "Class" column
        drop_cols: a list of strings with columns to drop from the returned X dataframe (other than "Class")

    Returns:
        an (X, y) tuple where X is a dataframe with features and Y is a series of class labels (0,1)
    """

    X = df[df.Class.isin([target_class, control_class])] # restrict to two classes
    y = X.Class.replace({control_class: 0, target_class: 1})  # recode class labels so 1 is positive/target, 0 is negative/control
    X = X.drop(["Class"]+drop_cols, axis=1) # drop Class and any other columns
    return X, y


def eval_model(model, X: pd.DataFrame, y: pd.Series, target_class: str, control_class: str, threshold: float = 0.5):
    """Prints a classification report, displays a ConfusionMatrix, and plots a ROC curve.

    Args:
        model: a fitted sklearn model object
        X: a dataframe of features
        y: a series of class labels
        target_class: a string specifying the name of the target class
        control_class: a string specifying the name of the negative/control class in the "Class" column
        threshold: the decision threshold (between 0 and 1; default 0.5).
    """

    # Classification report
    preds = (model.predict_proba(X)[:,1] >= threshold).astype(int) # get binary prediction for threshold
    print(f"Threshold for classification: {threshold}")
    print(classification_report(y, preds, target_names=[control_class, target_class]))
    print(f'N.B.: "recall" = sensitivity for the group in this row (e.g. {target_class}); specificity for the other group ({control_class})\n'\
          f'N.B.: "precision" = PPV for the group in this row (e.g. {target_class}); NPV for the other group ({control_class})\n')
    # Confusion matrix
    cm = confusion_matrix(y, preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[control_class, target_class])
    disp.plot(cmap='binary')
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:,1])
    thr_idx = (np.abs(thresholds - threshold)).argmin() # find index of value closest to chosen threshold
    _, ax = plt.subplots()
    plot_roc_curve(model, X, y, name = f"{target_class} vs. {control_class}", ax=ax)
    ymin, ymax = ax.get_ylim(); xmin, xmax = ax.get_xlim()
    ax.set_ylim(ymin, ymax);  ax.set_xlim(xmin, xmax)
    plt.vlines(x=fpr[thr_idx], ymin=ymin, ymax=tpr[thr_idx], color='k', linestyle='--', axes=ax) # plot line for fpr at threshold
    plt.hlines(y=tpr[thr_idx], xmin=xmin, xmax=fpr[thr_idx], color='k', linestyle='--', axes=ax) # plot line for tpr at threshold


def calc_roc_cv(classifier, cv, X, y):
    """
    Calculate true positive rates and AUCs for a cross-validated classifier
    adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    Args:
        classifier: a scikit learn model object
        cv: a sckitlearn cross-validation object
        X: a pandas dataframe or numpy array of features
        y: a pandas series or numpy array with class labels

    Returns:
        tprs: list of true positive rates (one for each cross-validation iteration)
        aucs: list of ROC AUCS (one for each cross-validation iteration)
    """

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    clf = clone(classifier) # clone to avoid modifying an already trained estimator

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.to_numpy()
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()

    for _, (train, test) in enumerate(cv.split(X, y)):
        model = clf.fit(X[train], y[train])
        y_score = model.predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

    return tprs, aucs


def plot_roc_cv(tprs: List[np.ndarray], aucs: List[np.ndarray], fig, ax, reuse: bool = False,
                fig_title: str = None, line_color: str = 'b', legend_label: str = 'Mean ROC'):
    """
    Plot the ROC curve for a cross-validated classifier
    adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html

    Args:
        tprs: output of calc_roc_cv()
        aucs: output of plot_roc_cv()
        fig: matplotlib figure handle to plot to
        ax: matplotlib axis handle to plot to
        reuse: boolean; whether the figure/axis already contains a prior ROC curve
        fig_title: string with figure title
        line_color: string with matplotlib color abbreviation or hex code
        legend_label: string with label for the legend

    Returns the matplotlib figure and axis objects
    """

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    if not reuse:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='k', alpha=.25)
        ax.set(title=fig_title,
               xlim=[-0.05, 1.05], xlabel='False positive rate',
               ylim=[-0.05, 1.05], ylabel='True positive rate')

    ax.plot(mean_fpr, mean_tpr, color=line_color,
        label=f"{legend_label} (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})",
        lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=line_color, alpha=.2)

    ax.legend(loc="lower right")
    return fig, ax


def main():

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold

    df = generate_data('imid')

    model = LogisticRegression(solver='liblinear')
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=40)

    # Extract features and class labels
    X,y = prep_data(df, 'SLE', 'BBD', drop_cols = ["Arthritis","Pleurisy","Pericarditis","Nefritis"] + ["dsDNA1"])

    model.fit(X,y)

    # Report model metrics
    # N.B. this uses whole dataset; should be separate testing set of course
    eval_model(model, X, y, 'SLE', 'BBD')

    # Plot ROC curve
    tprs, aucs = calc_roc_cv(model, cv, X, y)

    fig, ax = plt.subplots()
    plot_roc_cv(tprs, aucs, fig, ax)


if __name__ == '__main__':
    main()