import os
import pickle

import numpy as np
import pandas as pd
import torch as pt


def create_directory(directory: str = None):
    """Create a directory if it does not exist.

    Args:
        directory: path to the directory to create
    """
    try:
        os.makedirs(directory, exist_ok=False)
    except FileExistsError:
        print('Output directory exists')


def save_obj(obj, name):
    """Save a Python object to a pickle file.

    Args:
        obj: object to save
        name: file path (without .pkl extension)
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def check_if_file_exists(filename):
    """Check if a file exists on disk."""
    return os.path.isfile(filename)


def check_tensors_for_nans(directory: str,
                           file_extension: str = '.pt'):
    """Check if there are any NaNs in the tensors in a directory.

    Args:
        directory: directory containing the tensors
        file_extension: file extension to filter by

    Returns:
        DataFrame with columns ['file ID', 'NANs image', 'NANs mask']
    """
    nans_table = pd.DataFrame(columns=['file ID', 'NANs image', 'NANs mask'])
    for file in os.listdir(directory):
        if not file.endswith(file_extension):
            continue
        tensor = pt.load(directory + file)
        nans_image = np.sum(np.isnan(tensor[0]))
        nans_mask = np.sum(np.isnan(tensor[1]))
        file_id = file.split('/')[-1].split('.')[0]
        nans_table = pd.concat((nans_table,
                                pd.DataFrame(columns=['file ID', 'NANs image', 'NANs mask'],
                                             data=[[file_id, nans_image, nans_mask]])),
                               ignore_index=True)
    return nans_table
