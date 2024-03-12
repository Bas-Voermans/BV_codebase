import os
import pandas as pd
import numpy as np
import pickle

from joblib import Parallel, delayed
from tqdm import tqdm
import copy
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
from . import models

"""
██    ██ ████████ ██ ██      ███████     ███    ███  █████   ██████ ██   ██ ██ ███    ██ ███████       ██      ███████  █████  ██████  ███    ██ ██ ███    ██  ██████  
██    ██    ██    ██ ██      ██          ████  ████ ██   ██ ██      ██   ██ ██ ████   ██ ██            ██      ██      ██   ██ ██   ██ ████   ██ ██ ████   ██ ██       
██    ██    ██    ██ ██      ███████     ██ ████ ██ ███████ ██      ███████ ██ ██ ██  ██ █████   █████ ██      █████   ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███ 
██    ██    ██    ██ ██           ██     ██  ██  ██ ██   ██ ██      ██   ██ ██ ██  ██ ██ ██            ██      ██      ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██ 
 ██████     ██    ██ ███████ ███████     ██      ██ ██   ██  ██████ ██   ██ ██ ██   ████ ███████       ███████ ███████ ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████       

  CONTENTS:
 - perform_stability_runs_classification
 - perform_stability_runs_regression
 - perm_imp
 - get_perm_imp          
 - save_results_dict   
 - load_results_dict                                                                                                                                                                                                                                                                                                                                                                                                                                                   
"""

def perform_stability_runs_classification(
    X,  # Feature matrix of the dataset
    y,  # Label vector of the dataset
    test_size,  # Fraction of samples used for evaluation
    n_splits = 10,  # Number of stability runs to perform
    feature_names = None,
    method = models.tree, # Machine learning method
    method_params = {}, # Additional parameters for the ML method (look at specific docs for the method)
    return_importances = False, # Whether feature importances should be calculated or not
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
):
    """
    This function performs stability runs for classification tasks.
    It splits the data into training and testing sets multiple times,
    fits the model, makes predictions, and calculates scores.
    """
    # Define the keys for the results dictionary
    keys = [
        'train_scores',
        'val_scores',
        'test_scores',
        'y_train_val',
        'y_train_preds',
        'y_test',
        'y_preds',
        'feat_imps',
    ]

    # Define the keys for printing (these are train, val and test scores)
    printing_keys = keys[:3]

    # Initialize the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Initialize the counter
    i=0
    # Loop over the splits
    for train_val,test in sss.split(X,y):
        # Increment the counter
        i+=1

        # Split the data into training and testing sets
        X_train_val, y_train_val = X[train_val], y[train_val]
        X_test, y_test = X[test], y[test]

        # Fit the model and make predictions
        cur_results_dict = method(
            X_train_val, 
            y_train_val, 
            X_test, 
            y_test, 
            return_importances=return_importances, 
            n_jobs=n_jobs, 
            verbosity=verbosity, 
            **method_params
            )
        
        # If this is the first split, initialize the results dictionary
        if i == 1:
            results_dict = cur_results_dict
        # If this is the second split, convert the results to lists
        elif i == 2:
            for key in keys:
                results_dict[key] = [results_dict[key], cur_results_dict[key]]
        # If this is the third or later split, append the results to the lists
        else:
            for key in keys:
                results_dict[key].append(cur_results_dict[key])
    
        # If verbosity is greater than 0, print the results
        if verbosity > 0:
            print('shuffle',i,'done')
            for key in printing_keys:
                print(key+':',np.mean(results_dict[key]),'+\-',np.std(results_dict[key]))
            print()

    # add some input parameters to the results dictionairy
    results_dict['X'] = X
    results_dict['y'] = y
    results_dict['n_splits'] = n_splits

    
    if feature_names != None:
        results_dict['feature_names'] = feature_names
    else:
        results_dict['feature_names'] = range(np.shape(X)[1])

    # Return the results dictionary
    return results_dict

def perform_stability_runs_regression(
    X,  # Feature matrix of the dataset
    y,  # continuous vector of the dataset
    test_size,  # Fraction of samples used for evaluation
    n_splits = 10,  # Number of stability runs to perform
    n_quantiles = 3, # Number of quantiles to use for the stratifiedshufflesplit
    feature_names = None,
    method = models.tree, # Machine learning method
    method_params = {}, # Additional parameters for the ML method (look at specific docs for the method)
    return_importances = False, # Whether feature importances should be calculated or not
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
):
    """
    This function performs stability runs for classification tasks.
    It splits the data into training and testing sets multiple times,
    fits the model, makes predictions, and calculates scores.
    """

    # Define the keys for the results dictionary
    keys = [
        'train_scores',
        'val_scores',
        'test_scores',
        'y_train_preds',
        'y_preds',
        'feat_imps',
    ]
    
    # Define the keys for printing (these are train, val and test scores)
    printing_keys = keys[:3]

    # Initialize the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    # Initialize the counter
    i=0

    # Use pandas qcut function to split y into n_quantiles with labels ranging from 0 to n_quantiles-1
    y_quant = pd.qcut(y, n_quantiles, labels=False)

    # Loop over the splits
    for train_val,test in sss.split(X,y_quant):
        # Increment the counter
        i+=1

        # Split the data into training and testing sets
        X_train_val, y_train_val = X[train_val], y[train_val]
        X_test, y_test = X[test], y[test]

        # Fit the model and make predictions
        cur_results_dict = method(
            X_train_val, 
            y_train_val, 
            X_test, 
            y_test, 
            return_importances=return_importances, 
            n_jobs=n_jobs, 
            verbosity=verbosity, 
            **method_params
            )
        
        # If this is the first split, initialize the results dictionary
        if i == 1:
            results_dict = cur_results_dict
        # If this is the second split, convert the results to lists
        elif i == 2:
            for key in keys:
                results_dict[key] = [results_dict[key], cur_results_dict[key]]
        # If this is the third or later split, append the results to the lists
        else:
            for key in keys:
                results_dict[key].append(cur_results_dict[key])
    
        # If verbosity is greater than 0, print the results
        if verbosity > 0:
            print('shuffle',i,'done')
            for key in printing_keys:
                print(key+':',np.mean(results_dict[key]),'+\-',np.std(results_dict[key]))
            print()
    
    # add some input parameters to the results dictionairy
    results_dict['X'] = X
    results_dict['y'] = y
    results_dict['n_splits'] = n_splits

    if feature_names != None:
        results_dict['feature_names'] = feature_names
    else:
        results_dict['feature_names'] = range(np.shape(X)[1])

    # Return the results dictionary
    return results_dict


def perm_imp(feat_n, model, X, y, scoring_function, n_repeats=10, task='classify'):
    """
    This function calculates the permutation importance of a feature.
    
    Parameters:
    feat_n (int): The index of the feature for which the permutation importance is calculated.
    model (object): The trained model.
    X (numpy array): The input data.
    y (numpy array): The target data.
    scoring_function (function): The function used to score the model's predictions.
    n_repeats (int): The number of times to repeat the permutation and scoring.
    task (str): The type of task. Can be 'classify' or 'regression'.
    
    Returns:
    float: The mean permutation importance score over all repeats.
    """

    # Predict the target variable based on the task type
    if task == 'classify':
        y_pred = model.predict_proba(X)[:,1]
    elif task == 'regression':
        y_pred = model.predict(X)
    
    # Calculate the initial score
    score_0 = scoring_function(y, y_pred)
    
    # Initialize the list to store permutation scores
    perm_scores = []
    
    # Repeat the process for n_repeats times
    for i in range(n_repeats):
        # Create a deep copy of the input data
        X_perm = copy.deepcopy(X)
        # Permute the feature values
        X_perm[:, feat_n] = np.random.permutation(X_perm[:, feat_n])
        # Predict the target variable for the permuted data
        if task == 'classify':
            y_pred_perm = model.predict_proba(X_perm)[:,1]
        elif task == 'regression':
            y_pred_perm = model.predict(X_perm)
        
        # Calculate the absolute difference between the initial score and the score after permutation
        perm_scores.append(np.abs(score_0 - scoring_function(y, y_pred_perm)))
    
    # Return the mean permutation score
    return np.mean(perm_scores)




def get_perm_imp(model, X, y, scoring_function, n_repeats=10, task='classify', n_jobs=-1, verbosity=2):
    """
    This function calculates the permutation importance of all features.
    
    Parameters:
    model (object): The trained model.
    X (numpy array): The input data.
    y (numpy array): The target data.
    scoring_function (function): The function used to score the model's predictions.
    n_repeats (int): The number of times to repeat the permutation and scoring.
    task (str): The type of task. Can be 'classify' or 'regression'.
    n_jobs (int): The number of jobs to run in parallel.
    verbosity (int): The level of verbosity.
    
    Returns:
    list: The permutation importance scores for all features.
    """
    # Get the number of features
    number_of_features = np.shape(X)[1]
    
    # Print a message if verbosity is greater than 1
    if verbosity > 1:
        print('getting permutation importances')
    
    # Calculate the permutation importance for each feature in parallel
    if verbosity > 1:
        imps = Parallel(n_jobs=n_jobs)(delayed(perm_imp)(feat_n, model, X, y, scoring_function, n_repeats=n_repeats, task=task) for feat_n in tqdm(range(number_of_features)))
    else:
        imps = Parallel(n_jobs=n_jobs)(delayed(perm_imp)(feat_n, model, X, y, scoring_function, n_repeats=n_repeats, task=task) for feat_n in range(number_of_features))

    # Return the permutation importances
    return imps

def save_results_dict(filename, results_dict):
    """
    Save a dictionary to a pickle file.

    Parameters:
    filename (str): The name of the file to save the dictionary to.
    results_dict (dict): The dictionary to save.
    """
    # Open the file in write-binary mode
    if filename[-4:] == '.pkl':
        f = open(filename, "wb")
    else:
        f = open(filename + ".pkl", "wb")

    # Write the dictionary to the pickle file
    pickle.dump(results_dict, f)

    # Close the file
    f.close()

def load_results_dict(filename):
    """
    Load a dictionary from a pickle file.

    Parameters:
    filename (str): The name of the file to load the dictionary from.

    Returns:
    dict: The loaded dictionary.
    """
    # Open the file in read-binary mode
    file = open(filename, 'rb')

    # Load the dictionary from the pickle file
    data = pickle.load(file)

    # Close the file
    file.close()

    return data