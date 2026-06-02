import numpy as np
import pandas as pd
import re


def find_nearest(array: np.array, value: float):
    '''This is a function that returns the index and value of the nearest value in an array.'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def get_index_from_depth_range(depth: np.array, depth_range: list):
    depth_index = np.logical_and(depth >= np.min(depth_range), depth <= np.max(depth_range))
    return depth_index, np.where(depth_index)[0]


def get_index_from_start_depth(depth,
                               start_depth,
                               depth_range):
    start_index, value = find_nearest(depth, start_depth)
    end_index = start_index + depth_range
    all_indices = np.zeros((depth.shape[0], 1), dtype=bool)
    if (end_index > all_indices.shape[0]):
        end_index = all_indices.shape[0]
    all_indices[start_index:end_index] = True
    return all_indices.squeeze(), np.where(all_indices)[0]


def get_depth_only(data_path: str,
                   file_name: str):
    '''This is a function that returns the depth data from a LAS file.
    It returns the depth data and the line number where the data starts.'''
    first_dataline = get_first_data_linenumber(file_name, data_path)
    depth = pd.read_csv(data_path + file_name, skiprows=first_dataline, usecols=[0], header=None,
                        encoding='utf-8').to_numpy(
        dtype='float')
    return np.squeeze(depth), first_dataline


def grep(pattern: str,
         file: str):
    '''This function opens a file and searches for a pattern. If the pattern is found, the line number is returned.'''
    with open(file, 'r') as f:
        counter = 0
        for line in f:
            counter += 1
            if re.match(pattern, line):
                return counter


def get_first_data_linenumber(file_name: str,
                              data_path: str,
                              pattern: str = '~LOG_DATA | LOG_DEFINITION'):
    '''This function opens a file and looks for the first line that contains data'''
    # "~LOG_DATA"
    with open(data_path + file_name,
              encoding="utf8",
              errors='ignore') as f:
        for num, line in enumerate(f, 1):
            if pattern in line:
                return num + 1
        print('String not found in file')
        return None


# return int(str(check_output(["grep", "-n", pattern, self.data_path + file_name]), 'utf-8').split(':')[0]) + 1

def get_otv_data(data_path, file_name, image_columns):
    first_dataline = get_first_data_linenumber(file_name, data_path)
    file_in = pd.read_csv(data_path + file_name, skiprows=first_dataline, header=None)
    depth = file_in[0].to_numpy(dtype='float')
    data_array = np.ndarray((file_in.shape[0], file_in.shape[1] - 1, 3), dtype='uint8')
    for index, col in enumerate(image_columns):
        temp = file_in[index + 1].str.split('.', expand=True)
        data_array[:, col - 1, :] = temp.apply(lambda x: pd.to_numeric(x, downcast='unsigned')).to_numpy(
            dtype='uint8')
    return depth, data_array

def get_atv_data(data_path, file_name, image_columns):
    first_dataline = get_first_data_linenumber(file_name, data_path)
    file_in = pd.read_csv(data_path + file_name, skiprows=first_dataline, header=None)
    depth = file_in[0].to_numpy(dtype='float')
    data_array = np.ndarray((file_in.shape[0], file_in.shape[1] - 1, 3), dtype='uint8')
    for index, col in enumerate(image_columns):
        temp = file_in[index + 1].str.split('.', expand=True)
        data_array[:, col - 1, :] = temp.apply(lambda x: pd.to_numeric(x, downcast='unsigned')).to_numpy(
            dtype='uint8')
    return depth, data_array

def get_data_subset_from_start_depth(data_path,
                                     file_name,
                                     start_depth,
                                     depth_range,
                                     image_columns):
    depth, first_dataline = get_depth_only(data_path, file_name)
    depth_index, index_values = get_index_from_start_depth(depth, start_depth, depth_range)
    start_row = first_dataline + np.min(index_values)
    file_in = pd.read_csv(data_path + file_name,
                          skiprows=start_row,
                          nrows=np.sum(depth_index),
                          header=None,
                          encoding='utf-8')
    depth_from_file = file_in.iloc[:, 0].to_numpy()
    data_array = np.ndarray((file_in.shape[0], file_in.shape[1] - 1, 3), dtype='uint8')
    for index, col in enumerate(image_columns):
        temp = file_in[index + 1].str.split('.', expand=True)
        data_array[:, col, :] = temp.apply(lambda x: pd.to_numeric(x, downcast='unsigned')).to_numpy(dtype='uint8')
    return depth_from_file, data_array, depth_index


def get_data_subset_from_depth_range(file_name,
                                     depth_range,
                                     data_path: str,
                                     depth=None,
                                     data_type: str = 'otv',
                                     first_dataline=None):
    if (depth is None or first_dataline is None):
        depth, first_dataline = get_depth_only(data_path, file_name)
    depth_index, index_values = get_index_from_depth_range(depth, depth_range)
    start_row = first_dataline + np.min(index_values)
    file_in = pd.read_csv(data_path + file_name,
                          skiprows=start_row,
                          nrows=np.sum(depth_index),
                          header=None,
                          encoding='utf-8')
    #this is uncorrected depth
    # depth_from_file = file_in.iloc[:, 0].to_numpy()

    if data_type == 'otv':
        data_array = np.ndarray((file_in.shape[0], file_in.shape[1] - 1, 3), dtype='uint8')
        for index in range(file_in.shape[1] - 1):
            if data_type == 'otv':
                temp = file_in[index + 1].str.split('.', expand=True)
                data_array[:, index, :] = temp.apply(lambda x: pd.to_numeric(x, downcast='unsigned')).to_numpy(
                    dtype='uint8')
    elif data_type == 'atv':
        data_array = file_in.to_numpy(dtype='float')[:, 1:]
    else:
        raise ValueError('Data type not supported')
    return data_array, depth_index
