

import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score,root_mean_squared_error
from sklearn.model_selection import RepeatedStratifiedKFold,RepeatedKFold,GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor

from xgboost import XGBClassifier,XGBRegressor

"""
███    ███  ██████  ██████  ███████ ██      ███████ 
████  ████ ██    ██ ██   ██ ██      ██      ██      
██ ████ ██ ██    ██ ██   ██ █████   ██      ███████ 
██  ██  ██ ██    ██ ██   ██ ██      ██           ██ 
██      ██  ██████  ██████  ███████ ███████ ███████ 
"""

# Parameter grids for Extra Trees and XGBoost
param_grid_xtr = {
    "n_estimators":[ 100, 300, 500, 800, ],
    "max_depth":[2 ,3, 5, None],
    "min_samples_leaf":[1,5,7],
    "bootstrap": [False,True]
}

param_grid_xgb = {
    'max_depth': [2, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300, 800, 1000, ],
    'min_child_weight': [1, 5],
    'gamma': [0.5, 2],
    'subsample': [0.5, 0.6, 0.8],
    'colsample_bytree': [0.6, 0.8, 1.0],
}


def tree(
    X_train_val, y_train_val, X_test, y_test, 
    algo='xtr',
    n_cv_folds=3,
    n_cv_repeats=1,
    task='classify',
    scoring=None,
    param_grid=None,
    n_jobs=-1,
    verbosity=0,
    return_importances = False,
    imp_algo = None,
    imp_algo_params={},
        ):
    
    """
    This function performs a grid search over the specified parameter grid for the specified algorithm.
    It uses cross-validation and calculates scores for the training, validation, and test sets.
    """

    # Initialize a dictionary to store the scores from the training, validation, and testing phases
    results_dict = {
        'train_scores':[],
        'val_scores':[],
        'test_scores':[],
    }

    # Create a cross-validation strategy using Repeated Stratified K-Fold
    if task == 'classify':
        splitter = RepeatedStratifiedKFold(n_splits=n_cv_folds, n_repeats=n_cv_repeats)
    elif task == 'regression':
        splitter = RepeatedKFold(n_splits=n_cv_folds, n_repeats=n_cv_repeats)


    # If the algorithm is Extra Trees (xtr), initialize the model based on the task (classification or regression)
    if algo == 'xtr':
        if task == 'classify':
            model = ExtraTreesClassifier()
        if task == 'regression':
            model = ExtraTreesRegressor()
        if param_grid == None:
            param_grid = param_grid_xtr
        
    if algo == 'xgb':
        if task == 'classify':
            model = XGBClassifier()
        if task == 'regression':
            model = XGBRegressor()
        if param_grid == None:
            param_grid = param_grid_xgb



    # If the task is classification, set the scoring metric to ROC AUC, for regression set to RMSE
    if scoring == None:
        if task=='classify':
            scoring='roc_auc'
            scoring_function = roc_auc_score
        elif task == 'regression':
            scoring = 'neg_root_mean_squared_error'
            scoring_function=root_mean_squared_error



    # Initialize a Grid Search CV object with the model, parameter grid, scoring metric, cross-validation strategy, and other parameters
    search=GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=n_jobs,
        verbose=0,
        refit=True) 

    # Fit the Grid Search CV object to the training data and et the best model from the grid search
    search.fit(X_train_val,y_train_val)
    best_model = search.best_estimator_

    # Make predictions on the training and test data based on the task
    if task=='classify':
        y_train_preds = best_model.predict_proba(X_train_val)[:,1]
        y_preds = best_model.predict_proba(X_test)[:,1]

        out_scores = [
        scoring_function(y_train_val,y_train_preds),
        search.best_score_,
        scoring_function(y_test,y_preds)
        ]

    elif task == 'regression':
        y_train_preds = best_model.predict(X_train_val)
        y_preds = best_model.predict(X_test)
        
        out_scores = [
        scoring_function(y_train_val,y_train_preds),
        -search.best_score_,
        scoring_function(y_test,y_preds)
        ]



    # Add the scores to the results dictionary and print them if verbose > 1
    for i,key in enumerate(results_dict.keys()):
        results_dict[key].append(out_scores[i])
        if verbosity > 1:
            print(key+':',np.mean(results_dict[key]))
    if verbosity > 1:
        print()
    # Add the labels of the current shuffle to the results dictionary
    results_dict['y_train_val'] = y_train_val
    results_dict['y_test'] = y_test       

    # Add the model output to the results dictionary
    results_dict['y_train_preds'] = y_train_preds
    results_dict['y_preds'] = y_preds

    # Add the model parameters to the results dictionary
    results_dict['algo'] = algo
    results_dict['n_cv_folds'] = n_cv_folds
    results_dict['n_cv_repeats'] = n_cv_repeats
    results_dict['task'] = task
    results_dict['scoring'] = scoring





    # if required, add the feature importanes to the results dictionairy
    if return_importances:
        if imp_algo == None:
            feat_imps = best_model.feature_importances_
        else:
            feat_imps = imp_algo(best_model, X_test, y_test, scoring_function, task=task, verbosity=verbosity, **imp_algo_params)

        results_dict['feat_imps'] = feat_imps
    else:
        results_dict['feat_imps'] = None

    return results_dict