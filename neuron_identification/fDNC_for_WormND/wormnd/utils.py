import os 
import numpy as np 

def get_all_files(path_from):
    all_files = []
    for fold, sub_fold, files in os.walk(path_from):
        # print(fold)
        for file_name in files:
            file_path = os.path.join(fold, file_name)
            # print(file_path)
            all_files.append(file_path)
    return all_files

def min_max_normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val:
        # Avoid division by zero if all values in the array are the same
        return np.zeros_like(array)
    normalized_array = 2 * (array - min_val) / (max_val - min_val) - 1
    return normalized_array

def mean_std_normalize(array):
    mean_val = np.mean(array)
    std_val = np.std(array)
    if std_val == 0:
        # Avoid division by zero if all values in the array are the same
        return np.zeros_like(array)
    normalized_array =   (array - mean_val) / std_val
    return normalized_array

def center_sccale_normalize(array):
    # assert np.max(array)>100
    min_val = np.min(array)
    max_val = np.max(array)
    mean_val = np.median(array)
    if max_val == min_val:
        # Avoid division by zero if all values in the array are the same
        return np.zeros_like(array)
    normalized_array = (array - mean_val) / 84
    return normalized_array



# # import pandas as pd
# name =r"C:\Users\jd\Desktop\tr_te_val_splits.csv"
# name = "/scratch/jd4587/fDNC_Daniel/tr_te_val_splits.csv"
# tr_te_val_splits_df = pd.read_csv(name)
# # Separate data into train, validation, and test sets
# train_files = tr_te_val_splits_df[tr_te_val_splits_df['Group'] == 'train']['Filename'].tolist()
# validation_files = tr_te_val_splits_df[tr_te_val_splits_df['Group'] == 'validation']['Filename'].tolist()
# test_files = tr_te_val_splits_df[tr_te_val_splits_df['Group'] == 'test']['Filename'].tolist()
