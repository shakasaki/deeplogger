#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#labelling in MB3
# Necessary functions

import numpy as np
import pandas as pd

def filter_labels_from_range(label_df: pd.DataFrame,
                             filter_type: str,
                             filter_range: (list, tuple)):
    '''Filter the label dataframe based on entries of individual columns that have numeric values (e.g., depth and aperture).
    The dataframe returned has only the filtered values between the desired range.
    Args:
        label_df: label dataframe
        filter_type: column name of the dataframe to filter
        filter_range: range of values to filter
    Returns:
        label_df: filtered label dataframe

        '''
    # check if filter type exists in dataframe columns, otherwise return empty dataframe
    if filter_type not in label_df.columns:
        print('Filter type does not exist in dataframe columns')
        return pd.DataFrame()
    # check if filter range is a list or tuple with two entries, otherwise return empty dataframe
    if not isinstance(filter_range, (list, tuple)) or len(filter_range) != 2:
        print('Filter range must be a list or tuple with two entries')
        return pd.DataFrame()
    # filter based on filter type and range
    return label_df[(label_df[filter_type] >= filter_range[0]) & (label_df[filter_type] <= filter_range[1])].copy()


def find_nearest(array, value):
    """Find the nearest value in an array.
    Args:
        array: array to search
        value: value to search for
    Returns:
        idx: index of the nearest value in the array
            """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def crop_depth(Z_up, Z_down, depth):
    """Crop the depth of the picks to the depth of the borehole.
    Args:
        Z_up: depth of the top of the pick
        Z_down: depth of the bottom of the pick
        depth: depth vector of the borehole
    Returns:
        Z_up: depth of the top of the pick cropped to the depth of the borehole
        Z_down: depth of the bottom of the pick cropped to the depth of the borehole
        """
    max_depth = depth[-2]
    min_depth = depth[2]
    Z_up[Z_up > max_depth] = max_depth
    Z_down[Z_down > max_depth] = max_depth
    Z_up[Z_up < min_depth] = min_depth
    Z_down[Z_down < min_depth] = min_depth
    return Z_up, Z_down


def get_label(azimuth: float,
              dip: float,
              depth: float,
              aperture: float,
              depth_vector: np.array,
              bh_diameter: float,
              azimuth_values: int = 360):
    """Get a label for a pick in the form of a rasterized image.
    Args:
        azimuth: azimuth of the pick in degrees
        dip: dip of the pick in degrees
        depth: mean depth of the pick
        aperture: aperture of the pick in mm
        depth_vector: depth vector of the borehole
        bh_diameter: borehole diameter in meters
        azimuth_values: number of azimuth values to use for the pick
    Returns:
        im: rasterized image of the pick
        """
    # convert aperture to meters
    aperture = aperture / 1000
    # if mean depth is outside the depth vector, then the pick is not valid
    azimuth_radians = np.linspace(0, 2 * np.pi, azimuth_values)
    azimuth_range = range(azimuth_values)
    amplitude = depth + np.sin(azimuth_radians + np.deg2rad(azimuth + 90)) * np.tan(np.deg2rad(dip)) * bh_diameter / 2
    Z_up = amplitude + aperture / 2
    Z_down = amplitude - aperture / 2
    Z_up, Z_down = crop_depth(Z_up, Z_down, depth_vector)
    im = np.zeros((depth_vector.shape[0], azimuth_values))
    if np.array(Z_up == Z_down).all():
        im[np.digitize(Z_up, depth_vector), azimuth_range] = 1
    else:
        Z_up_M = np.tile(Z_up, (depth_vector.shape[0], 1))
        Z_down_M = np.tile(Z_down, (depth_vector.shape[0], 1))
        Z_M = np.tile(depth_vector, (azimuth_values, 1)).T
        im[np.logical_and(Z_M < Z_up_M, Z_M > Z_down_M)] = 1
    return im

def get_pick_coordinates(alpha,
                         beta,
                         mean_depth,
                         depth_vector,
                         aperture: float = 1.0,
                         bh_diameter: float = 0.1,
                         azimuth_values: int = 360,):
    azimuth_degrees = np.linspace(0, 2 * np.pi, azimuth_values) + np.pi
    amplitude = np.tan(np.deg2rad(90) - alpha) * bh_diameter/2
    #        Z = -amplitude*np.cos(azimuth) + mean_depth
    Z = amplitude * np.cos(azimuth_degrees + beta) + mean_depth
    Z_up = Z + aperture / 2
    Z_down = Z - aperture / 2
    Z_up, Z_down = crop_depth(Z_up, Z_down, depth_vector)
    depth_index = np.arange(0, depth_vector.shape[0])
    depth_index[np.digitize(Z_up, depth_vector)]
    pixeled_line_up = depth_index[np.digitize(Z_up, depth_vector)]
    pixeled_line_down = depth_index[np.digitize(Z_down, depth_vector)]
    depth_pixels = np.concatenate((pixeled_line_up, np.flip(pixeled_line_down)))
    #        angle_pixels = np.concatenate((azimuth_degrees,np.flip(azimuth_degrees)))
    angle_pixels = np.concatenate((np.flip(azimuth_degrees), azimuth_degrees))
    coordinates = list(zip(angle_pixels, depth_pixels))
    return coordinates

def get_polygons(label_df, depth_vector):
    from shapely import Polygon
    polygons = list()
    polygon_colors = list()
    for index, label in enumerate(label_df):
#        polygon_colors.append(get_color(label[4]))
        print('Accessing label # ' + str(index))
        coordinates = digitize_label(label, depth_vector, index)
        polygons.append(Polygon(coordinates))
    return polygons

def raster_map_from_label(polygons, n_rows, n_cols):
    from rasterio import features
    return features.rasterize(polygons, (n_rows, n_cols), all_touched=True)


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    return kernel_2D


def convolution(image, kernel, average=False):
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height,
    pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output


def gaussian_blur(image, kernel_size):
    kernel = gaussian_kernel(kernel_size, sigma=np.sqrt(kernel_size))
    conv = convolution(image, kernel, average=False)
    return conv/conv.max()


# In[ ]:


from deeplogger.utils.download_data import *   
import os
# assign the "path_to_data" the path where MB3 was already downloaded       
from deeplogger import DATA_DIR

path_to_data = DATA_DIR + "MB3" + os.sep
# go into a directory and make a list of the files that are in there
files = os.listdir(path_to_data)
files.sort()
print(files)


# In[ ]:


from deeplogger.importLASv3 import *
from deeplogger.image_processing import *
file_name = files[8]
depth_subset= get_depth_only(file_name=files[8],
               data_path=path_to_data)

depth_subset, data_subset, index = get_data_subset_from_depth_range(file_name=files[8],
                                                                    data_path=path_to_data,
                                                                    data_type='otv',
                                                                    depth_range=[118, 120])
data_subset = replace_empty_measurements(data_subset, 0)


# In[ ]:


#import the data and select a range of depth (amplitude)
file_atv_am =files[9]
depth_subset_ATV_am= get_depth_only(file_name=file_atv_am, #you can also just put the number of the data location in the files list, which is "files[10]" for the amplitude MB3 and "files[5]" for the travel time MB3
               data_path=path_to_data)

depth_subset_ATV_am, data_subset_ATV_am, index = get_data_subset_from_depth_range(file_name=file_atv_am,
                                                                    data_path=path_to_data,
                                                                    data_type='atv',
                                                                    depth_range=[118, 120])
#take-off the zero values within the data subset
data_subset_ATV_am = replace_empty_measurements(data_subset_ATV_am, 0)

depth_vector_atv = depth_subset_ATV_am


# In[ ]:


label = get_label(azimuth=60,
                  dip=75,
                  depth=118.80,
                  aperture=1000,
                  depth_vector=depth_vector_atv,
                  bh_diameter=0.1,
                  azimuth_values=360)


# In[ ]:


#depth correction applied to the OTV data
depth_subset, data_subset, index = get_data_subset_from_depth_range(file_name=files[8],
                                                                    data_path=path_to_data,
                                                                    data_type='otv',
                                                                    depth_range=[118.5, 120.5])
# tries plot with the filtered ATV image (SVD)
data_subset_svd_am = data_subset_svd_am, svd_decomp_am = remove_svd(data_subset_ATV_am, low_s=0, high_s=2)

#tries for plotting
import numpy as np
import matplotlib.pyplot as plt
data_subset_svd_am = replace_empty_measurements(data_subset_svd_am, 0)
# %% Plot the results and compare with a blurred mask
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()
axs[0].imshow(data_subset_ATV_am, aspect='auto', extent=[0, 359, 120, 118], cmap='hot')
axs[0].imshow(label,
                 cmap='hsv',
                 aspect='auto',
                 extent=[0, 359, depth_vector_atv.max(), depth_vector_atv.min()],
                 alpha=0.3)

axs[0].set_xlabel('Azimuth [°]')
axs[0].set_ylabel('Depth [m]')
axs[0].set_title('Raw label on ATV filtered image')

# Blur the label with a gaussian kernel

label_blurred = gaussian_blur(label, kernel_size=5)
axs[1].imshow(data_subset_ATV_am, aspect='auto', extent=[0, 359, 120, 118], cmap ='hot')
axs[1].imshow(label_blurred,
                 cmap='hsv',
                 aspect='auto',
                 extent=[0, 359, depth_vector_atv.max(), depth_vector_atv.min()],
                 alpha=0.3)
axs[1].set_xlabel('Azimuth [°]')
axs[1].set_title('Blurred label (kernel=5) on ATV filtered image')

# Plot data_subset_atv with blurred label
label_blurred_atv = gaussian_blur(label, kernel_size=5)

axs[2].imshow(data_subset, aspect='auto', extent=[0, 359, 120, 118], cmap='hot')
axs[2].imshow(label_blurred_atv,
                 cmap='hsv',
                 aspect='auto',
                 extent=[0, 359, depth_vector_atv.max(), depth_vector_atv.min()],
                 alpha=0.3)
axs[2].set_title(
    'blurred label on raw OTV  image')


axs[3].imshow(label,
                 cmap='cividis',
                 aspect='auto',
                 extent=[0, 359, depth_vector_atv.max(), 
                         depth_vector_atv.min()],
                 alpha=1)
axs[3].set_title('Label only')


# Plot a column of the image before and after blurring with transparency
#axs[1, 1].plot(depth_vector_atv, label[:, 180], label='Original', alpha=1, color='black')
#axs[1, 1].plot(depth_vector_atv, label_blurred[:, 180], label='Blurred', alpha=0.5, color='red')
#axs[1, 1].legend()
#axs[1, 1].set_xlabel('Depth [m]')
#axs[1, 1].set_ylabel('Probability')
#axs[1, 1].set_xlim(118, 120)
plt.show()

#here, the depth correction is of +0.5 for the OTV, -0.5 m for the ATV




