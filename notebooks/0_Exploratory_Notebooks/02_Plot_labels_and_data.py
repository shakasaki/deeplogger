# import the necessary packages
from deeplogger.utils.download_data import *
from deeplogger.importLASv3 import get_data_subset_from_depth_range
import pandas as pd

path_to_metadata = DATA_DIR + 'MB3' + os.sep
path_to_picks = DATA_DIR + 'MB3' + os.sep
path_to_metadata = download_VALTER_metadata()
path_to_picks = download_VALTER_picks()
print(path_to_metadata)
print(path_to_picks)

# load the xlsx file with pandas
metadata = pd.read_table(path_to_metadata, header=0, delimiter=',')
picks = pd.read_table(path_to_picks, header=0, delimiter=',')

# print the first 5 rows of the picks
from deeplogger.common_helpers import (filter_dataframe_from_text,
                                       filter_labels_from_range)

# example: filter the picks to only get the ones for MB3
picks_MB3 = filter_dataframe_from_text(picks,
                                       'Borehole',
                                       'MB3')
print(picks_MB3.head())

from deeplogger import DATA_DIR
import os

path_to_data = DATA_DIR + 'MB3' + os.sep  # the os.sep in the end adds a / or \ depending on the operating system
print(path_to_data)

from deeplogger.importLASv3 import get_depth_only

# Load the data from MB3
# if it is already downloaded, this command can be used to find the data on the computer)
# path_to_data, files = download_VALTER_borehole('MB3')
print(path_to_data)

# list files in directory
files = os.listdir(path_to_data)

file_name_MB3 = path_to_data + files[10]  # amplitude data which goes to 120m depth
print(file_name_MB3)  # verify the name of the file and its date
# file_name_MB3 = files [5] #travel time data which goes to 120m depth
depth_MB3, values = get_depth_only(file_name=files[10],
                                   data_path=path_to_data)

# print original number of picks before filtering
print('Number of picks before filtering: ', picks.shape[0])
# filter the picks from the depth range of the MB3 dataset
picks_MB3 = filter_labels_from_range(picks, 'Depth (m)',
                                     [depth_MB3.min(), depth_MB3.max()])
# number of picks after filtering
print('Number of picks after filtering: ', picks_MB3.shape[0])

# plot the picks as a rasterized image (to be used for training later on)
import numpy as np


def crop_depth(Z_up, Z_down, depth):
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
                azimuth_values: int):
    # convert aperture to meters
    aperture = aperture / 1000
    # if mean depth is outside the depth vector, then the pick is not valid
    if depth < depth_vector.min() or depth > depth_vector.max():
        return None
    azimuth_radians = np.linspace(0, 2 * np.pi, azimuth_values)
    azimuth_range = range(azimuth_values)
    amplitude = depth + np.sin(azimuth_radians + np.deg2rad(azimuth + 90)) * np.tan(np.deg2rad(dip)) * bh_diameter / 2
    #np.tan(np.deg2rad(dip))*np.sin(np.deg2rad(90 + azimuth + azimuth_range)) * bh_diameter / 2
    Z_up = amplitude + aperture / 2
    Z_down = amplitude - aperture / 2
    Z_up, Z_down = crop_depth(Z_up, Z_down, depth_vector)
    depth_index = np.arange(0, depth_vector.shape[0])
    depth_index[np.digitize(Z_up, depth_vector)]
    pixeled_line_up = depth_index[np.digitize(Z_up, depth_vector)]
    pixeled_line_down = depth_index[np.digitize(Z_down, depth_vector)]
    depth_pixels = np.concatenate((pixeled_line_up, np.flip(pixeled_line_down)))
    #        angle_pixels = np.concatenate((azimuth_degrees,np.flip(azimuth_degrees)))
    angle_pixels = np.concatenate((np.flip(azimuth_range), azimuth_range))
    coordinates = np.array(list(zip(angle_pixels, depth_pixels)))
    im = np.zeros((depth_vector.shape[0], azimuth_values))
    im[coordinates[:,1], coordinates[:,0]] = 1
    return im



def get_pick_coordinates(alpha,
                         beta,
                         mean_depth,
                         aperture,
                         depth_vector,
                         bh_diameter,
                         azimuth_values):
    # convert aperture to meters
    aperture = aperture / 1000
    # if mean depth is outside the depth vector, then the pick is not valid
    if mean_depth < depth_vector.min() or mean_depth > depth_vector.max():
        return None
    azimuth_radians = np.linspace(0, 2 * np.pi, azimuth_values)
    azimuth_degrees = range(azimuth_values)
    amplitude = np.tan(np.deg2rad(90) - alpha) * bh_diameter / 2
    #        Z = -amplitude*np.cos(azimuth) + mean_depth
    Z = amplitude * np.cos(azimuth_radians + beta) + mean_depth
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
    im = np.zeros((depth_vector.shape[0], azimuth_values))
    im[coordinates[:,1], coordinates[:,0]] = 1
    return im


def get_polygons(label_df, depth_vector,
                 bh_diameter: float = 0.1,
                 min_aperture: float = 1):
    from shapely import Polygon
    polygons = list()
    for index, label in label_df.iterrows():
        #        polygon_colors.append(get_color(label[4]))
        # check if aperture is zero and assign to minimum aperture
        if label['Aperture (mm)'] == 0:
            label['Aperture (mm)'] = min_aperture
        coordinates = get_pick_coordinates(label['Azimuth (deg)'],
                                           label['Dip (deg)'],
                                           mean_depth=label['Depth (m)'],
                                           depth_vector=depth_vector,
                                           aperture=label['Aperture (mm)'],
                                           bh_diameter=bh_diameter,
                                           azimuth_values=360
                                           )
        polygons.append(Polygon(coordinates))
    return polygons


def raster_map_from_label(polygons, n_rows, n_cols):
    from rasterio import features
    return features.rasterize(polygons,
                              (n_rows, n_cols),
                              fill=0,
                              all_touched=True)


# polygons = get_polygons(picks_MB3, depth_MB3)
import matplotlib.pyplot as plt

# raster = raster_map_from_label(polygons, 360, 360)
# plt.imshow(raster, cmap='gray')

# np.sum(raster, 0)
# plt.show()

depth_subset_MB3, data_subset_MB3, index = get_data_subset_from_depth_range(file_name=files[10],
                                                                            data_path=path_to_data,
                                                                            data_type='atv',
                                                                            depth_range=[140, 142])
# remove the nan values
from deeplogger.image_processing import replace_empty_measurements

data_subset_MB3 = replace_empty_measurements(data_subset_MB3, 0)

plt.imshow(data_subset_MB3)


# label = get_label(azimuth=220,
#                   dip = 70,
#                   depth=30,
#                   aperture=0.001,
#                   depth_vector=np.linspace(29,31,100),
#                   bh_diameter=0.1,
#                   azimuth_values=360)

# image = np.zeros((depth_vector.shape[0], azimuth))
# image[coordinates[:, 0], coordinates[:, 1]] = 1


azimuth = 220
dip = 70
depth = 30
aperture = 1
depth_vector = np.linspace(29,31,100)
azimuth_values = 360
bh_diameter = 0.1
# convert aperture to meters
# if mean depth is outside the depth vector, then the pick is not valid
azimuth_radians = np.linspace(0, 2 * np.pi, azimuth_values)
azimuth_range = range(azimuth_values)
amplitude = depth + np.sin(azimuth_radians + np.deg2rad(azimuth + 90)) * np.tan(np.deg2rad(dip)) * bh_diameter / 2
#np.tan(np.deg2rad(dip))*np.sin(np.deg2rad(90 + azimuth + azimuth_range)) * bh_diameter / 2
Z_up = amplitude + aperture / 2
Z_down = amplitude - aperture / 2
Z_up, Z_down = crop_depth(Z_up, Z_down, depth_vector)
depth_index = np.arange(0, depth_vector.shape[0])
depth_index[np.digitize(Z_up, depth_vector)]
pixeled_line_up = depth_index[np.digitize(Z_up, depth_vector)]
pixeled_line_down = depth_index[np.digitize(Z_down, depth_vector)]
depth_pixels = np.concatenate((pixeled_line_up, np.flip(pixeled_line_down)))
#        angle_pixels = np.concatenate((azimuth_degrees,np.flip(azimuth_degrees)))
angle_pixels = np.concatenate((np.flip(azimuth_range), azimuth_range))
coordinates = np.array(list(zip(angle_pixels, depth_pixels)))

im = np.zeros((depth_vector.shape[0], azimuth_values))
im[coordinates[:,1], coordinates[:,0]] = 1

plt.imshow(im,
           extent=[0, 360, depth_vector.min(), depth_vector.max()],
)