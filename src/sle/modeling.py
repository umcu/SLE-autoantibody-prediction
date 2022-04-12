from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, plot_roc_curve

def prep_data(df: pd.DataFrame, target_class: str, control_class: str, drop_cols: list = []) -> Tuple[pd.DataFrame, pd.Series]:
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
        threshold: the decision threshold (between 0 and 1; default 0.5)
    
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
    fig, ax = plt.subplots()
    plot_roc_curve(model, X, y, name = f"{target_class} vs. {control_class}", ax=ax)
    ymin, ymax = ax.get_ylim(); xmin, xmax = ax.get_xlim()
    ax.set_ylim(ymin, ymax);  ax.set_xlim(xmin, xmax)
    plt.vlines(x=fpr[thr_idx], ymin=ymin, ymax=tpr[thr_idx], color='k', linestyle='--', axes=ax) # plot line for fpr at threshold
    plt.hlines(y=tpr[thr_idx], xmin=xmin, xmax=fpr[thr_idx], color='k', linestyle='--', axes=ax) # plot line for tpr at threshold