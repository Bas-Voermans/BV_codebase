import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score,roc_curve
from statannotations.Annotator import Annotator
from scipy import stats
import statsmodels.api as sm

from matplotlib import rcParams
import matplotlib

import sys
sys.path.append('C:\\Users\\basvo\\Documents\\GitHub\\BV_codebase')

import machine_learning

rcParams['font.weight'] = 'bold'

color_p = ['#ca0020',
'#f4a582',
'#f7f7f7',
'#92c5de',
'#0571b0'][::-1]

color_b = [color_p[0],color_p[-1]]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_p)

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
- boxplot_with_test
- perm_test_boxplot
- corrmap_top_feats
- scatter_plot
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
    fig = plt.figure(figsize=(10, 8))
    b=sns.barplot(x=imps_plotting, y=feature_names_plotting, hue=hue_plotting, palette=color_b)
    b.legend_.remove()
    plt.xlabel(u'abs( Δscore )', weight='bold')

    return fig


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
    fig = plt.figure(figsize=(10, 8))
    ax = sns.barplot(x=imps_plotting, y=feature_names_plotting, hue=directions_plotting, palette=color_b)
    plt.axvline(0, linestyle='--', color='black')
    h, l = ax.get_legend_handles_labels()
    labels = ['Higher in class 0', 'Higher in class 1']

    ax.legend(h, labels, title="Direction of importance")
    plt.xlabel(u'abs( Δscore )', weight='bold')

    return fig

def boxplot_with_test(df, x_var, y_var = 'twl', test='t-test_ind', ax = None, pvalue=None):
    """
    Generate a boxplot with t-test annotations for each unique value in the specified x variable.

    Parameters:
    df (pandas.DataFrame): The input dataframe.
    x_var (str): The name of the column to be used for the x-axis.

    Returns:
    None
    """
    
    # Create a new DataFrame with x_var and y_var columns
    boxplot_data = pd.DataFrame({
        y_var: df[y_var],
        x_var: df[x_var]
    })

    #drop the nans in the data
    nancount = np.sum(np.sum(np.isnan(boxplot_data)))

    if nancount > 0:
        print(f'{nancount} NaN values detected in the data. Dropping rows with NaN values.')
        boxplot_data.dropna(axis=0, inplace=True)

    if ax == None:
        ax = sns.boxplot(data=boxplot_data, x=x_var, y=y_var, palette=color_b, showfliers=False)
    else:
        sns.boxplot(data=boxplot_data, x=x_var, y=y_var, ax=ax, palette=color_b, showfliers=False)
        
    # Add annotations with t-test results
    tmp = [tuple(boxplot_data[x_var].unique())]
    annot = Annotator(ax, tmp, data=boxplot_data, x=x_var, y=y_var,)
    annot.configure(test=test)
    annot.apply_test()
    ax, test_results = annot.annotate()

    # # Add horizontal line at y=50 (response treshold)
    # ax.axhline(35, linestyle='--', color='black')

    # Set labels and formatting
    ax.set_ylabel(y_var, weight='bold')
    ax.set_xlabel(x_var, weight='bold')
    
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plt.savefig('../../../figures/weightloss_boxplots/twl_'+x_var+'.pdf')

    if ax == None:
        fig = plt.gcf()
        plt.show()
        return fig
    else:
        return ax

def perm_test_boxplot(results_dict_true, results_dict_perm, task, omic, save=False, ax=None):
    if save:
        filename_out = 'figures/'+omic+'_'+task+'.pdf'

    scores_true = pd.DataFrame(results_dict_true['test_scores'])
    scores_true.columns = [task+' score']
    scores_true[omic+' run type'] = 'true '+task

    scores_perm = pd.DataFrame(results_dict_perm['test_scores'])
    scores_perm.columns = [task+' score']
    scores_perm[omic+' run type'] = 'permuted '+task

    scores_df = pd.concat([scores_true, scores_perm], axis=0)

    if ax == None:
        fig = boxplot_with_test(scores_df, omic+' run type', task+' score')
    else:
        ax = boxplot_with_test(scores_df, omic+' run type', task+' score', ax=ax)
        return ax
    if save:
        fig.savefig(filename_out)

def corrmap_top_feats(results_dict, topN=10):

    top_feats = machine_learning.utils.return_top_feats(results_dict, top_n=topN)
    X_top = machine_learning.utils.get_X_from_results_dict(results_dict)[top_feats]

    corrs,_ = stats.spearmanr(X_top)
    corrs = pd.DataFrame(corrs, index = top_feats, columns = top_feats)

    ax = sns.clustermap(corrs,cmap='RdBu', vmin=-1,vmax=1)

    fig = plt.gcf()
    plt.show()
    return fig

def scatter_plot(df, x_var, y_var, p_value=None):
    # Create a new DataFrame with x_var and y_var columns
    boxplot_data = pd.DataFrame({
        y_var: df[y_var],
        x_var: df[x_var]
    })


    nancount = np.sum(np.sum(np.isnan(boxplot_data)))
    
    if nancount > 0:
        print(f'{nancount} NaN values detected in the data. Dropping rows with NaN values.')

    boxplot_data.dropna(axis=0, inplace=True)

    # Plot boxplots for each unique value in the x_var column
    ax = sns.scatterplot(data=boxplot_data, x=x_var, y=y_var)

    # Calculate linear regression and RMSE
    x = boxplot_data[x_var].values.reshape(-1, 1)
    y = boxplot_data[y_var].values.reshape(-1, 1)
    X = sm.add_constant(x)
    linear_regression = sm.OLS(y, X).fit()
    y_pred = linear_regression.predict(X)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    # Add linear trendline
    ax.plot(x, y_pred, color='black', linestyle='-')

    # Calculate Spearman correlation coefficient and p-value
    if p_value == None:
        spearman_rho, p_value = stats.spearmanr(boxplot_data[x_var], boxplot_data[y_var])
    else:
        spearman_rho, _ = stats.spearmanr(boxplot_data[x_var], boxplot_data[y_var])
    

    # Set labels and formatting
    ax.set_ylabel(y_var)
    ax.set_xlabel(x_var)
    
    if p_value > 0.0001:
        ax.set_title(f"{y_var} vs. {x_var} \n (Spearman rho={spearman_rho:.3f}, p={p_value:.7f})\nRMSE={rmse:.3f}")
    else:
        ax.set_title(f"{y_var} vs. {x_var} \n (Spearman rho={spearman_rho:.3f}, p={p_value:.3e})\nRMSE={rmse:.3f}")


    # plt.savefig('../../../figures/weightloss_correlations/adjusted_twl_'+x_var+'.pdf')

    fig = plt.gcf()
    # plt.show()
    return fig