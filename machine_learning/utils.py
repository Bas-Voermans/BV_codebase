import os
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm
import copy
from sklearn.model_selection import StratifiedShuffleSplit
from . import models

def perform_stability_runs_classification(
    X,  #feature matrix of the dataset
    y,  #label vector of the dataset
    test_size,  #fraction of samples used for evaluation
    n_splits = 10,  #number of stability runs to perform
    method = models.tree, #machine learning method
    method_params = {}, #additional parameters for the ml method (look at specific docs for the method)
    return_importances = False, #whether feature importances should be calculated or not
    n_jobs=-1, #number of parallelization jobs to use
    verbosity=1, #level of verbosity {0 : no printing whatsoever, 1 : running averages of performance, 2 : running average + run scores}
):
    keys = [
        'train_scores',
        'val_scores',
        'test_scores',
        'y_train_preds',
        'y_preds',
        'feat_imps',
    ]
    printing_keys = keys[:3]

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    i=0
    for train_val,test in sss.split(X,y):
        i+=1

        X_train_val, y_train_val = X[train_val], y[train_val]
        X_test, y_test = X[test], y[test]

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
        
        if i == 1:
            results_dict = cur_results_dict
        elif i == 2:
            for key in keys:
                results_dict[key] = [results_dict[key], cur_results_dict[key]]
        else:
            for key in keys:
                results_dict[key].append(cur_results_dict[key])
    
        if verbosity > 0:
            print('shuffle',i,'done')
            for key in printing_keys:
                print(key+':',np.mean(results_dict[key]),'+\-',np.std(results_dict[key]))
            print()

    return results_dict

def perm_imp(feat_n,model,X,y,scoring_function,n_repeats=10,task='classify'):
    if task == 'classify':
        y_pred = model.predict_proba(X)[:,1]
    elif task == 'regression':
        y_pred = model.predict(X)
    
    score_0 = scoring_function(y,y_pred)  
    perm_scores = []
    for i in range(n_repeats):
        X_perm = copy.deepcopy(X)
        X_perm[:,feat_n] = np.random.permutation(X_perm[:,feat_n])
        if task == 'classify':
            y_pred_perm = model.predict_proba(X_perm)[:,1]
        elif task == 'regression':
            y_pred_perm = model.predict(X_perm)
        
        perm_scores.append(np.abs(score_0 - scoring_function(y,y_pred_perm)))
    return np.mean(perm_scores)

def get_perm_imp(
    model,
    X,
    y,
    scoring_function,
    n_repeats=10,
    task='classify',
    n_jobs=-1,
    verbosity=2,
):  
    print(n_repeats)
    
    number_of_features = np.shape(X)[1]
    if verbosity>0:
        print('getting permutation importances')
        imps = Parallel(n_jobs=n_jobs)(delayed(perm_imp)(feat_n,model,X,y,scoring_function, n_repeats=n_repeats,task=task,) for feat_n in tqdm(range(number_of_features)))
    else:
        imps = Parallel(n_jobs=n_jobs)(delayed(perm_imp)(
            feat_n,
            model,
            X,
            y,
            scoring_function, 
            n_repeats=n_repeats,
            task=task,
            ) for feat_n in range(number_of_features))

    return imps