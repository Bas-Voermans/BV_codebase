
import pandas as pd
import numpy as np

"""
██    ██ ████████ ██ ██      ███████     ███    ███ ███████ ████████  █████   ██████  ███████ ███    ██  ██████  ███    ███ ██  ██████ ███████ 
██    ██    ██    ██ ██      ██          ████  ████ ██         ██    ██   ██ ██       ██      ████   ██ ██    ██ ████  ████ ██ ██      ██      
██    ██    ██    ██ ██      ███████     ██ ████ ██ █████      ██    ███████ ██   ███ █████   ██ ██  ██ ██    ██ ██ ████ ██ ██ ██      ███████ 
██    ██    ██    ██ ██           ██     ██  ██  ██ ██         ██    ██   ██ ██    ██ ██      ██  ██ ██ ██    ██ ██  ██  ██ ██ ██           ██ 
 ██████     ██    ██ ███████ ███████     ██      ██ ███████    ██    ██   ██  ██████  ███████ ██   ████  ██████  ██      ██ ██  ██████ ███████ 


 CONTENTS:
 - remove_all_zero_species
 - abundance_selection
 - sparsity_selection
 - process_species_names
                                                                                                                                                                                                                                                                                     
"""

def remove_all_zero_species(df):
    """
    This function removes all columns (species) in the dataframe that only contain zeros.
    
    Parameters:
    df (DataFrame): The input dataframe.
    
    Returns:
    DataFrame: The dataframe with all-zero columns removed.
    """
    # Calculate the number of samples (rows) in the dataframe
    n_samples = len(df)
    
    # Identify the columns that only contain zeros
    all_zero_cols = np.sum(df==0.0,axis=0) == n_samples
    
    # Identify the columns that contain non-zero values
    kept_columns = np.sum(df==0.0,axis=0) != n_samples

    # Create a new dataframe that only includes the columns with non-zero values
    df_out = df.loc[:,kept_columns]

    # Print the number of columns removed
    print(f'there were {np.sum(all_zero_cols)} removed from the dataset')
    
    # If any columns were removed, print their names
    if np.sum(all_zero_cols) > 0 :
        removed_columns = df.loc[:,all_zero_cols]
        print('removed columns:', list(removed_columns.columns))
    
    return df_out

def abundance_selection(df,to_select):
    """
    This function selects the top 'to_select' columns based on their total abundance.
    
    Parameters:
    df (DataFrame): The input dataframe.
    to_select (int or float): The number or fraction of columns to select.
    
    Returns:
    DataFrame: The dataframe with only the selected columns.
    """
    # Determine the number of columns to select
    if type(to_select) == int:
        n_to_select = to_select
    if type(to_select) == float:
        n_to_select = int(np.round(len(df.columns)*to_select))

    # Calculate the total abundance of each column
    summed_abundances = np.sum(df, axis=0)

    # Get the indices of the columns sorted by their total abundance in descending order
    locs = np.argsort(summed_abundances)[::-1]
    
    # Select the top 'n_to_select' columns
    to_keep = locs.iloc[:n_to_select]

    # Create a new dataframe that only includes the selected columns
    df_out = df.iloc[:,to_keep]

    return df_out

def sparsity_selection(df,minimum_non_zero_fraction):
    """
    This function removes columns that have a fraction of non-zero values less than 'minimum_non_zero_fraction'.
    
    Parameters:
    df (DataFrame): The input dataframe.
    minimum_non_zero_fraction (float): The minimum fraction of non-zero values a column must have to be kept.
    
    Returns:
    DataFrame: The dataframe with sparse columns removed.
    """
    # Calculate the fraction of non-zero values in each column
    non_zero_fractions = np.sum(df > 0, axis=0)/len(df.columns)

    # Identify the columns that have a fraction of non-zero values greater than 'minimum_non_zero_fraction'
    kept_columns = non_zero_fractions>minimum_non_zero_fraction

    # Create a new dataframe that only includes the non-sparse columns
    df_out = df.loc[:,kept_columns]

    return df_out

def process_species_names(feature_names, delim='__', idx=-1):
    """
    This function processes a list of species names by splitting each name at a specified delimiter and 
    appending a specified part of the split name to a new list.

    Parameters:
    feature_names (list): A list of strings representing species names.
    delim (str, optional): The delimiter at which to split each name. Defaults to '__'.
    idx (int, optional): The index of the split name to append to the new list. Defaults to -1, which corresponds to the last part of the split name.

    Returns:
    list: A new list containing the processed species names.
    """
    # Initialize the list to store the processed species names
    n_feat_names = []

    # Iterate over the input list of species names
    for feature_name in feature_names:
        # Split the current name at the specified delimiter
        feature_name = feature_name.split(delim)
        # Append the specified part of the split name to the new list
        n_feat_names.append(feature_name[idx])
    
    # Return the new list of processed species names
    return n_feat_names