import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import roc_auc_score,roc_curve

"""
██████  ██       ██████  ████████ ████████ ██ ███    ██  ██████  
██   ██ ██      ██    ██    ██       ██    ██ ████   ██ ██       
██████  ██      ██    ██    ██       ██    ██ ██ ██  ██ ██   ███ 
██      ██      ██    ██    ██       ██    ██ ██  ██ ██ ██    ██ 
██      ███████  ██████     ██       ██    ██ ██   ████  ██████  

CONTENTS"
- auc_curves
- imps_simple_barplot
- imps_directional_barplot
"""

def auc_curves(results_dict):
    """
    This function plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC).
    It also plots the standard deviation as a shaded area.

    Parameters:
    y_true (list of arrays): A list of arrays containing the true binary labels.
    y_pred (list of arrays): A list of arrays containing the predicted probabilities.

    Returns:
    None
    """
    y_true = results_dict['y_test']
    y_pred = results_dict['y_preds']

    
    # Get the number of ROC curves to be plotted
    n_curves = len(y_true)

    # Initialize the list to store the False Positive Rates (FPRs) for the plot
    fprs_plot = []

    # Initialize the lists to store the AUCs, FPRs, and True Positive Rates (TPRs)
    aucs = []
    fprs = []
    tprs = []

    # Calculate the AUC, FPR, and TPR for each set of true labels and predicted probabilities
    for i in range(n_curves):
        aucs.append(roc_auc_score(y_true[i], y_pred[i]))
        cur_fprs, cur_tprs, _ = roc_curve(y_true[i], y_pred[i])
        fprs.append(cur_fprs)
        tprs.append(cur_tprs)
        fprs_plot.extend(cur_fprs)

    # Get the unique FPRs for the plot
    fprs_plot = np.unique(fprs_plot)

    # Initialize the list to store the TPRs for the plot
    tprs_plot = []

    # Interpolate the TPRs at the FPRs for the plot
    for i in range(n_curves):
        cur_fprs, cur_tprs = fprs[i], tprs[i]
        cur_tprs_plot = np.interp(fprs_plot, cur_fprs, cur_tprs)
        cur_tprs_plot[0] = 0
        tprs_plot.append(cur_tprs_plot)

    # Calculate the mean and standard deviation of the TPRs
    mean_tprs = np.mean(tprs_plot, axis=0)
    sd_tprs = np.std(tprs_plot, axis=0)

    # Plot the ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fprs_plot, mean_tprs, color='black', lw=2, label=f'Mean ROC curve (AUC = {np.mean(aucs):.2f} $\pm$ {np.std(aucs):.2f})')

    # Plot the standard deviation area
    tprs_upper = np.minimum(mean_tprs + sd_tprs, 1)
    tprs_lower = np.maximum(mean_tprs - sd_tprs, 0)
    plt.fill_between(fprs_plot, tprs_lower, tprs_upper, color='grey', label=r'$\pm$ 1 std. dev.')

    # Add a red dotted reference line for an ROC AUC of 0.5
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red', label='Chance', alpha=.8)

    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')

    # Show the plot
    plt.show()

def imps_simple_barplot(results_dict, top_n=25):
    """
    This function plots a simple bar plot of the feature importances.

    Parameters:
    results_dict (dict): A dictionary containing feature names and their importances.
    top_n (int, optional): The number of top features to plot. Defaults to 25.
    """
    # Extract feature names and importances from the results dictionary
    feature_names = np.asarray(results_dict['feature_names'])
    imps = np.asarray(results_dict['feat_imps'])

    # Calculate the mean importances and get the indices that would sort the importances
    mean_imps = np.mean(results_dict['feat_imps'], axis=0)
    sorted_idx = np.argsort(mean_imps)[::-1]

    # Sort the feature names and importances
    feature_names = feature_names[sorted_idx]
    imps = imps[:, sorted_idx]

    # Get the top n feature names and importances
    feature_names_top_n = feature_names[:top_n]
    imps_top_n = imps[:, :top_n]

    # Prepare the data for plotting
    feature_names_plotting = np.asarray(list(feature_names_top_n) * results_dict['n_splits'])
    imps_plotting = imps_top_n.ravel()
    hue_plotting = np.ones_like(imps_plotting) * -1

    # Plot the bar plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=imps_plotting, y=feature_names_plotting, hue=hue_plotting, palette='vlag', legend=False)
    plt.xlabel(u'abs( Δ AUC )')


def imps_directional_barplot(results_dict, top_n=25):
    """
    This function plots a directional bar plot of the feature importances.

    Parameters:
    results_dict (dict): A dictionary containing feature names and their importances.
    top_n (int, optional): The number of top features to plot. Defaults to 25.
    """
    # Extract feature names, importances, X, and y from the results dictionary
    feature_names = np.asarray(results_dict['feature_names'])
    imps = np.asarray(results_dict['feat_imps'])
    X = results_dict['X']
    y = results_dict['y']

    # Calculate the mean importances and get the indices that would sort the importances
    mean_imps = np.mean(results_dict['feat_imps'], axis=0)
    sorted_idx = np.argsort(mean_imps)[::-1]

    # Sort the feature names and importances
    feature_names = feature_names[sorted_idx]
    imps = imps[:, sorted_idx]

    # Separate X into positive and negative classes based on y
    pos_X = X[np.where(y == 1)]
    neg_X = X[np.where(y == 0)]

    # Calculate the mean of pos_X and neg_X
    mean_pos_X = np.mean(pos_X, axis=0)
    mean_neg_X = np.mean(neg_X, axis=0)

    # Determine the direction of importance for each feature
    directions = []
    for i in range(len(feature_names)):
        if mean_pos_X[i] < mean_neg_X[i]:
            imps[:, i] *= -1
            directions.append(-1)
        else:
            directions.append(1)

    # Get the top n feature names, importances, and directions
    feature_names_top_n = feature_names[:top_n]
    imps_top_n = imps[:, :top_n]
    directions_top_n = directions[:top_n]

    # Prepare the data for plotting
    feature_names_plotting = np.asarray(list(feature_names_top_n) * results_dict['n_splits'])
    directions_plotting = np.asarray(directions_top_n * results_dict['n_splits'])
    imps_plotting = imps_top_n.ravel()

    # Plot the bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=imps_plotting, y=feature_names_plotting, hue=directions_plotting, palette='vlag')
    plt.axvline(0, linestyle='--', color='black')
    h, l = ax.get_legend_handles_labels()
    labels = ['Higher in class 0', 'Higher in class 1']

    ax.legend(h, labels, title="Direction of importance")
    plt.xlabel(u'abs( Δ AUC )')