import os
import pandas as pd
import numpy as np
import pickle
import random

from joblib import Parallel, delayed
from tqdm import tqdm
import copy
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit,RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score,root_mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
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
    selection_params = None, # Additional parameters for the feature selection method
    return_importances = False, # Whether feature importances should be calculated or not
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
    impute = False,
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

    if selection_params != None:
        keys.append('kept_feats')

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

                #check X for Nans, if permute is true, encode the data otherwise return an error
        if np.isnan(X_train_val).any() or np.isnan(X_test).any():
            if impute:
                imp_mean = IterativeImputer()
                imp_mean.fit(X_train_val)
                X_train_val = imp_mean.transform(X_train_val)
                X_test = imp_mean.transform(X_test)
                
            else:
                raise ValueError('X contains NaNs, please remove them or set impute to True')
        
        if selection_params != None:
            if selection_params['method'] == 'uni':
                selector = SelectKBest(f_classif, k=selection_params['n_to_keep'])
                selector.fit(X_train_val, y_train_val)
                X_train_val = selector.transform(X_train_val)
                X_test = selector.transform(X_test)
                to_keep = selector.get_support()

            if selection_params['method'] == 'rfe':
                to_keep = recurrent_feature_selection(X_train_val, y_train_val, selection_params['n_to_keep'],verbosity=verbosity)
                X_train_val = X_train_val[:,to_keep]
                X_test = X_test[:,to_keep]

            else:
                raise ValueError('method not recognized')
            
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
        
        if selection_params != None:
            kept_feats = feature_names[to_keep]
            cur_results_dict['kept_feats'] = kept_feats


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

    if len(feature_names) != 0:
        results_dict['feature_names'] = feature_names
    else:
        results_dict['feature_names'] = range(np.shape(X)[1])

    # Return the results dictionary
    return results_dict

def perform_permutation_test_classification(
    X,  # Feature matrix of the dataset
    y,  # Label vector of the dataset
    test_size,  # Fraction of samples used for evaluation
    n_splits = 10,  # Number of stability runs to perform
    feature_names = np.empty(0),
    method = models.tree, # Machine learning method
    method_params = {}, # Additional parameters for the ML method (look at specific docs for the method)
    selection_params = None, # Additional parameters for the feature selection method
    return_importances = False, # Whether feature importances should be calculated or not
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
    impute = False,
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

    if selection_params != None:
        keys.append('kept_feats')


    # Define the keys for printing (these are train, val and test scores)
    printing_keys = keys[:3]

    for i in range(n_splits):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

        # SHUFFLE THE Y COLUMN!
        y_rand = copy.deepcopy(y)
        np.random.shuffle(y_rand)
        y = copy.deepcopy(y_rand)


        # Loop over the splits
        for train_val,test in sss.split(X,y):
            # Increment the counter
            i+=1
            # Split the data into training and testing sets
            X_train_val, y_train_val = X[train_val], y[train_val]
            X_test, y_test = X[test], y[test]

            if np.isnan(X_train_val).any() or np.isnan(X_test).any():
                if impute:
                    imp_mean = IterativeImputer()
                    imp_mean.fit(X_train_val)
                    X_train_val = imp_mean.transform(X_train_val)
                    X_test = imp_mean.transform(X_test)
                else:
                    raise ValueError('X contains NaNs, please remove them or set impute to True')
            
            if selection_params != None:
                if selection_params['method'] == 'uni':
                    selector = SelectKBest(f_classif, k=selection_params['n_to_keep'])
                    selector.fit(X_train_val, y_train_val)
                    X_train_val = selector.transform(X_train_val)
                    X_test = selector.transform(X_test)

                    to_keep = selector.get_support()

                if selection_params['method'] == 'rfe':
                    to_keep = recurrent_feature_selection(X_train_val, y_train_val, selection_params['n_to_keep'],verbosity=verbosity)
                    X_train_val = X_train_val[:,to_keep]
                    X_test = X_test[:,to_keep]

                else:
                    raise ValueError('method not recognized')
                
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
            
            if selection_params != None:
                kept_feats = feature_names[to_keep]
                cur_results_dict['kept_feats'] = kept_feats

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

    
    if len(feature_names) != 0:
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
    selection_params = None, # Additional parameters for the feature selection method
    return_importances = False, # Whether feature importances should be calculated or not
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
    impute = False,
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
        'y_test',
    ]
    
    if selection_params != None and selection_params['type'] != 'decomposition':
        keys.append('kept_feats')
        
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
        
        
        if np.isnan(X_train_val).any() or np.isnan(X_test).any():
                if impute:
                    imp_mean = IterativeImputer(max_iter=1000)
                    imp_mean.fit(X_train_val)
                    X_train_val = imp_mean.transform(X_train_val)
                    X_test = imp_mean.transform(X_test)
                else:
                    raise ValueError('X contains NaNs, please remove them or set impute to True')
            
        if selection_params != None:
            if selection_params['method'] == 'uni':
                selector = SelectKBest(
                    f_regression, 
                    k=selection_params['n_to_keep'],
                )
                
                selector.fit(X_train_val, y_train_val)
                X_train_val = selector.transform(X_train_val)
                X_test = selector.transform(X_test)

                to_keep = selector.get_support()

            elif selection_params['method'] == 'rfe':
                to_keep = recurrent_feature_selection(
                    X_train_val, 
                    y_train_val, 
                    method_params['task'],
                    root_mean_squared_error,
                    verbosity=verbosity,
                    n_quantiles=n_quantiles,
                )
                
                X_train_val = X_train_val[:,to_keep]
                X_test = X_test[:,to_keep]
            
            elif selection_params['method'] == 'pca':
                n_components = selection_params['n_to_keep']
                pca = PCA(n_components=n_components)
                pca.fit(X_train_val)
                X_train_val = pca.transform(X_train_val)
                X_test = pca.transform(X_test)
            
            elif selection_params['method'] == 'pls':
                n_components = selection_params['n_to_keep']
                pls = PLSRegression(n_components=n_components)
                pls.fit(X_train_val, y_train_val)
                X_train_val = pls.transform(X_train_val)
                X_test = pls.transform(X_test)
            
            else:
                print(selection_params['method'])
                raise ValueError('method not recognized')
        # print(np.shape(X_train_val)[1])
        # print(len(to_keep))

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
        
        if selection_params != None and selection_params['type'] != 'decomposition':
            kept_feats = np.asarray(feature_names)[to_keep]
            cur_results_dict['kept_feats'] = kept_feats

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

    if len(feature_names) != 0:
        results_dict['feature_names'] = feature_names
    else:
        results_dict['feature_names'] = range(np.shape(X)[1])

    # Return the results dictionary
    return results_dict


def perform_permutation_test_regression(
    X,  # Feature matrix of the dataset
    y,  # continuous vector of the dataset
    test_size,  # Fraction of samples used for evaluation
    n_splits = 10,  # Number of stability runs to perform
    n_quantiles = 3, # Number of quantiles to use for the stratifiedshufflesplit
    feature_names = None,
    method = models.tree, # Machine learning method
    method_params = {}, # Additional parameters for the ML method (look at specific docs for the method)
    selection_params = None, # Additional parameters for the feature selection method
    return_importances = False, # Whether feature importances should be calculated or not
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
    impute = False,
    ):
    """
    This function performs stability runs for classification tasks.
    It splits the data into training and testing sets multiple times,
    fits the model, makes predictions, and calculates scores.
    """

    # Define the keys for printing (these are train, val and test scores)
    printing_keys = [
        'train_scores',
        'val_scores',
        'test_scores',
    ]

    # Initialize the StratifiedShuffleSplit object

    for i in range(n_splits):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)

        # Initialize the counter
        # i=0

        # SHUFFLE THE Y COLUMN!
        y_rand = copy.deepcopy(y)
        np.random.shuffle(y_rand)
        y = copy.deepcopy(y_rand)

        # Use pandas qcut function to split y into n_quantiles with labels ranging from 0 to n_quantiles-1
        y_quant = pd.qcut(y, n_quantiles, labels=False)

        # Loop over the splits
        for train_val,test in sss.split(X,y_quant):
            # Increment the counter
            i+=1

            # Split the data into training and testing sets
            X_train_val, y_train_val = X[train_val], y[train_val]
            X_test, y_test = X[test], y[test]
            
            if np.isnan(X_train_val).any() or np.isnan(X_test).any():
                if impute:
                    imp_mean = IterativeImputer()
                    imp_mean.fit(X_train_val)
                    X_train_val = imp_mean.transform(X_train_val)
                    X_test = imp_mean.transform(X_test)
                else:
                    raise ValueError('X contains NaNs, please remove them or set impute to True')
            
            if selection_params != None:
                if selection_params['method'] == 'uni':
                    selector = SelectKBest(
                        f_regression, 
                        k=selection_params['n_to_keep'],
                    )
                    
                    selector.fit(X_train_val, y_train_val)
                    X_train_val = selector.transform(X_train_val)
                    X_test = selector.transform(X_test)

                    to_keep = selector.get_support()

                if selection_params['method'] == 'rfe':
                    to_keep = recurrent_feature_selection(
                        X_train_val, 
                        y_train_val, 
                        method_params['task'],
                        root_mean_squared_error,
                        verbosity=verbosity,
                        n_quantiles=n_quantiles,
                    )
                    
                    X_train_val = X_train_val[:,to_keep]
                    X_test = X_test[:,to_keep]

                else:
                    raise ValueError('method not recognized')
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
                for key in results_dict.keys():
                    results_dict[key] = [results_dict[key], cur_results_dict[key]]
            # If this is the third or later split, append the results to the lists
            else:
                for key in results_dict.keys():
                    results_dict[key].append(cur_results_dict[key])
        
            # If verbosity is greater than 0, print the results
            if verbosity > 0:
                print('shuffle',i,'done')
                for key in printing_keys:
                    print(key+':',np.mean(results_dict[key]),'+\-',np.std(results_dict[key]))
                print()
    
    # add some input parameters to the results dictionairy
    # results_dict['X'] = X
    # results_dict['y'] = y
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

def return_top_feats(results_dict, top_n=25):

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

    return feature_names_top_n

def get_X_from_results_dict(results_dict):

    X = results_dict['X']
    
    feat_names = results_dict['feature_names']

    X = pd.DataFrame(X, columns=feat_names)

    return X

def feature_selection_sweep(
    X,  # Feature matrix of the dataset
    y,  # Label vector of the dataset
    test_size,  # Fraction of samples used for evaluation
    n_splits = 3,  # Number of stability runs to perform
    feature_names = None,
    method = models.tree, # Machine learning method
    method_params = {}, # Additional parameters for the ML method (look at specific docs for the method)
    n_jobs=-1, # Number of parallelization jobs to use
    verbosity=1, # Level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
    impute = False,
    selection_method = None,
):

 

    # Initialize the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    n_features = np.shape(X)[1]

    results_dict = {
        'n_feats_to_keep':[],
        'train_scores':[],
        'val_scores':[],
        'test_scores':[],
    }

    for n_feats_to_keep in range(1,n_features+1):
        
        # Define the keys for the results dictionary
        keys = [
            'train_scores',
            'val_scores',
            'test_scores',
        ]

        # Define the keys for printing (these are train, val and test scores)
        printing_keys = keys[:3]
        
        
        # Initialize the counter
        i=0
        # Loop over the splits
        for train_val,test in sss.split(X,y):
            # Increment the counter
            i+=1

            # Split the data into training and testing sets
            X_train_val, y_train_val = X[train_val], y[train_val]
            X_test, y_test = X[test], y[test]

                    #check X for Nans, if permute is true, encode the data otherwise return an error
            if np.isnan(X_train_val).any() or np.isnan(X_test).any():
                if impute:
                    imp_mean = IterativeImputer()
                    imp_mean.fit(X_train_val)
                    X_train_val = imp_mean.transform(X_train_val)
                    X_test = imp_mean.transform(X_test)
                else:
                    raise ValueError('X contains NaNs, please remove them or set impute to True')
                
            
            selector = SelectKBest(f_classif, k=n_feats_to_keep)
            selector.fit(X_train_val, y_train_val)
            X_train_val = selector.transform(X_train_val)
            X_test = selector.transform(X_test)

            # Fit the model and make predictions
            cur_results_dict = method(
                X_train_val, 
                y_train_val, 
                X_test, 
                y_test, 
                return_importances=False,
                n_jobs=n_jobs, 
                verbosity=verbosity, 
                **method_params
                )
            
            # If this is the first split, initialize the results dictionary
            if i == 1:
                n_results_dict = cur_results_dict
            # If this is the second split, convert the results to lists
            elif i == 2:
                for key in keys:
                    n_results_dict[key] = [n_results_dict[key], cur_results_dict[key]]
            # If this is the third or later split, append the results to the lists
            else:
                for key in keys:
                    n_results_dict[key].append(cur_results_dict[key])
        
        results_dict['n_feats_to_keep'].append(n_feats_to_keep)
        for key in printing_keys:
            results_dict[key].append(np.mean(n_results_dict[key]))

        
        # If verbosity is greater than 0, print the results
        if verbosity > 0:
            print('shuffle',i,'done')
            for key in printing_keys:
                print(key+':',np.mean(n_results_dict[key]),'+\-',np.std(n_results_dict[key]))
            print()


    results_dict['best_n_feats'] = results_dict['n_feats_to_keep'][np.argmax(results_dict['test_scores'])]
    results_dict['best_test_score'] = results_dict['test_scores'][np.argmax(results_dict['test_scores'])]

    # Return the results dictionary
    return results_dict

def rfe_feat_run(train, val, X, y, method, task, scoring_function, imp_algo=get_perm_imp):
    X_train, y_train = X[train], y[train]
    X_val, y_val = X[val], y[val]

    if method == 'xtr':
        if task == 'classify':
            model = ExtraTreesClassifier(n_jobs=-1)
        elif task == 'regression':
            model = ExtraTreesRegressor(n_jobs=-1)
        else:
            raise ValueError('task not recognized')
    else:
        raise ValueError('method not recognized')

    model.fit(X_train, y_train)

    #calculate aucs on the validation set
    if method == 'xtr':
        if task == 'classify':
            y_preds = model.predict_proba(X_val)[:,1]
        elif task == 'regression':
            y_preds = model.predict(X_val)
    
    auc = scoring_function(y_val, y_preds)

    #calculate feature importances
    if imp_algo == None:
        imp = model.feature_importances_
    else:
        imp = np.asarray(imp_algo(model, X_val, y_val, scoring_function, task=task, verbosity=0))
    return (auc, imp)

def recurrent_feature_selection(X, y, task, scoring_function, n_folds=3, n_repeats=10, method='xtr', n_quantiles=3, imp_algo=get_perm_imp, verbosity=2):

    if verbosity > 1:
        print('Starting recurrent feature selection')

    n_features = np.shape(X)[1]
    n_jobs = n_folds*n_repeats

    param_grid={}
    
    avg_aucs = []
    X_kept = []
    kept_feats_list = []
    
    #create a list of the number of features to keep
    n_to_keep_list = range(1,n_features)
    #reverse the list
    n_to_keep_list = n_to_keep_list[::-1]

    #start with all features (this is important for later when reading out the performances)
    kept_feats_list.append(list(range(n_features)))

    if task == 'classify':
        y_quant = y
    elif task=='regression':
        y_quant = pd.qcut(y, n_quantiles, labels=False)
    else:
        raise ValueError('task not recognized')

    for i in tqdm(n_to_keep_list):
        fold_splitter = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats)
        cur_aucs = []
        
        out = Parallel(n_jobs=n_jobs)(delayed(rfe_feat_run)(
            train, 
            val, 
            X, 
            y, 
            method, 
            task, 
            scoring_function, 
            imp_algo,
        ) for train, val in fold_splitter.split(X, y_quant))
        
        auc = [x[0] for x in out]
        imp = [x[1] for x in out]

        avg_auc = np.mean(auc)
        avg_aucs.append(avg_auc)
        X_kept.append(X)

        imp = np.mean(imp, axis=0)
        
        #get the indices of the n_to_keep best features
        sorted_idx = np.argsort(imp)[::-1]
        kept_feat_idx = sorted_idx[:i]

        X = X[:,kept_feat_idx]
        kept_feats_list.append(list(kept_feat_idx))
    X_kept.append(X)

    if task == 'classify':
        argmax = np.argmax(avg_aucs)
    elif task == 'regression':
        argmax = np.argmin(avg_aucs)
    
    best_n_to_keep = n_to_keep_list[argmax]
    best_auc = avg_aucs[argmax]
    best_X = X_kept[argmax]
    best_feats = kept_feats_list[argmax]

    if verbosity > 1:
        print('Best number of features to keep:',best_n_to_keep+1)
        print('Best AUC:',best_auc)
        print(best_feats)
    
    

    return best_feats