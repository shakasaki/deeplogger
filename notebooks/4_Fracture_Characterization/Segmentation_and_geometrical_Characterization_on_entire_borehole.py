import os
import torch
import pandas as pd

from deeplogger.model_architectures_ATV import UNetOTV
import matplotlib.pyplot as plt
import torch.utils.data as data
from deeplogger import DATA_DIR
from torch.nn import BCELoss
from matplotlib.colors import ListedColormap
import math
from skimage.transform import hough_line, hough_line_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

import pickle
import numpy as np
import cv2
import sys
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
import mplstereonet
import matplotlib.patches as mpatches
from matplotlib.table import Table
import matplotlib.markers



class Dataset_np(data.Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        image_id = self.list_IDs[index]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load your pre-processed image data (sample[0])
        sample = torch.load(image_id)
        image_data = sample[0]  # Assuming sample[0] contains the image data

        # Convert image data to a PyTorch tensor
        image = torch.tensor(image_data, device=device).float()

        return image, image_id

def extract_contours(segmentation_mask):
    # Find contours in the binary segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return contours

def separate_masks(segmentation_mask, original_image_shape, depth_vector):
    """
    Function to separate masks from a single concatenated mask image and adapt y-values to the depth.

    Parameters:
    - segmentation_mask: numpy array of shape (H, W) representing the mask.
    - original_image_shape: tuple representing the shape of the original image.
    - depth_vector: numpy array representing depth values for each y-coordinate.

    Returns:
    - individual_masks: List of separate masks.
    - mask_x_values_list: List of x coordinates of contours.
    - mask_y_values_list: List of y coordinates adapted to depth.
    """
    # Ensure segmentation_mask is a numpy array
    if isinstance(segmentation_mask, torch.Tensor):
        segmentation_mask = segmentation_mask.numpy()
    if not isinstance(segmentation_mask, np.ndarray):
        raise ValueError("segmentation_mask must be a numpy array.")
    if segmentation_mask.ndim != 2:
        raise ValueError("segmentation_mask must be a 2D array.")

    # Ensure segmentation_mask is of type uint8
    if segmentation_mask.dtype != np.uint8:
        segmentation_mask = segmentation_mask.astype(np.uint8)

    # Find contours in the binary segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # Initialize lists to store individual masks and coordinates
    individual_masks = []
    mask_x_values_list = []
    mask_y_values_list = []

    # Create blank images with the same shape as the original image
    for contour in contours:
        mask = np.zeros(original_image_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)  # Draw the contour
        individual_masks.append(mask)

        # Extract x and y values from the contour
        x_values = contour[:, 0, 0]
        y_values = contour[:, 0, 1]

        # Adapt y_values to the corresponding depth using depth_vector
        y_depth_values = depth_vector[y_values].flatten()

        mask_x_values_list.append(x_values)
        mask_y_values_list.append(y_depth_values)

    return individual_masks, mask_x_values_list, mask_y_values_list

def sinusoidal(x, A, B, C, D):
    # Ensure that x is an array
    x = np.array(x)
    B = 2 * np.pi / 360
    return A * np.sin(B * x + C) + D


def calculate_tangent_sinus(x, A, B, C, D, dx):
    y = sinusoidal(x, A, B, C, D)
    y_prime = ((sinusoidal(x + dx / 2, A, B, C, D) - sinusoidal(x - dx / 2, A, B, C, D)) / dx)
    y_double_prime = ((sinusoidal(x + dx / 2, A, B, C, D) - 2 * y + sinusoidal(x - dx / 2, A, B, C, D)) / (
            dx / 2) ** 2)

    # Calculate curvature
    curvature = np.abs(y_double_prime) / ((1 + y_prime ** 2) ** (3 / 2))

    # Adjust dx based on curvature
    # This factor can be tuned according to your specific needs
    adjustment_factor = factor  # It is defined when calling the function below (~line 360)

    dx_adjusted = dx * (1 + adjustment_factor / (1 + curvature))

    # Recalculate y' with adjusted dx
    dy = ((sinusoidal(x + dx_adjusted / 2, A, B, C, D) - sinusoidal(x - dx_adjusted / 2, A, B, C,
                                                                    D)) / dx_adjusted)
    slope = dy / dx_adjusted

    return slope, y, x, curvature


def calculate_perpendicular_slope(slope):
    return -1 / slope if slope != 0 else float('inf')
    # if the slope is 0, the perpendicular slope is infinite

# Function to find the closest point on the data to the tangent line

def find_closest_points(x_values, y_values, x_tangent, y_tangent, perp_slope):
    upper_closest_point = None
    lower_closest_point = None
    min_left_distance = float('inf')
    min_right_distance = float('inf')
    tolerance = 0.5  # Define a tolerance for floating-point comparison

    for i in range(len(x_values)):
        x2 = x_values[i]
        y2 = y_values[i]
        y_perp_line = perp_slope * (x2 - x_tangent) + y_tangent

        # Calculate the distance to the perpendicular line
        distance = np.abs(
            (perp_slope * x2 - y2 + y_tangent - perp_slope * x_tangent) / np.sqrt(perp_slope ** 2))

        if np.abs(y2 - y_perp_line) < tolerance:
            # Check if the point lies on the perpendicular line within the tolerance
            if x2 < x_tangent:
                # Point is on the left side
                if distance < min_left_distance:
                    min_left_distance = distance
                    upper_closest_point = (x2, y2)
            else:
                # Point is on the right side
                if distance < min_right_distance:
                    min_right_distance = distance
                    lower_closest_point = (x2, y2)

    return upper_closest_point, lower_closest_point

def calculate_beta_angle_vector(x_sin, x_value, x_sin_along_all_axis, y_sin_along_all_axis, D):
    # Find the index of x_sin where y_sin is closest to D
    distances = np.abs(y_sin_along_all_axis - D)
    crossing_index = np.argmin(distances)

    # Create an initial beta angle vector
    beta_angle_vector_along_all_axis = np.zeros(len(x_sin_along_all_axis))
    # at the crossing index, the beta angle is 0, from there, for each index, create a beta angle of unit of 1 and a step of 1. Do the same for negative decreasing values from 0 index to crossing index, so the 0 is maintained at the crossing index
    beta_angle_vector_along_all_axis = np.arange(0 - crossing_index, len(x_sin_along_all_axis) - crossing_index, 1)

    # Find the min and max values of x_sin
    min_x_value = np.min(x_value)
    # find the closest value of x_sin_along_all_axis to the min_x_sin
    closest_x_value = x_sin_along_all_axis[np.abs(x_sin_along_all_axis - min_x_value).argmin()]
    # determine the index of the x_sin_along_all_axis corresponding to the closest x_sin
    index_closest_x_value = np.where(x_sin_along_all_axis == closest_x_value)
    # do the same for the max value of x_sin
    max_x_value = np.max(x_value)
    closest_x_value_max = x_sin_along_all_axis[np.abs(x_sin_along_all_axis - max_x_value).argmin()]
    index_closest_x_value_max = np.where(x_sin_along_all_axis == closest_x_value_max)

    # cut off and only select the values between the min and max x_sin
    beta_angle_vector = beta_angle_vector_along_all_axis[
                        index_closest_x_value[0][0]:index_closest_x_value_max[0][0]]
    min_beta_angle = np.min(beta_angle_vector)
    max_beta_angle = np.max(beta_angle_vector)
    # Ensure the beta angle vector has the same length as x_sin
    length_cut_vector = len(beta_angle_vector_along_all_axis)
    length_x_sin = len(x_sin)

    if length_cut_vector < length_x_sin:
        # Calculate the step for adding missing values

        step = (max_beta_angle - min_beta_angle) / length_x_sin
        beta_angle_vector = np.arange(min_beta_angle, max_beta_angle, step)
    return beta_angle_vector

borehole_name = 'BFE_A_05'
# Load the model saved model
# Load the model saved model
model_name = '2D_unet_model07_16with_almost_allATV_handmade_labels75_epochs'
model_file = '2D_unet_model07_16with_almost_allATV_handmade_labels75_epochs.pt'
model_path = '/home/pperritaz/git/deeplogger/output/Bedretto_models' + os.sep + model_file

in_channels = 1  # Number of input channels (RGB image)
out_channels = 1  # Number of output channels
init_features = 32  # Initial number of features

# Instantiate the model
model = UNetOTV(in_channels=in_channels, out_channels=out_channels, init_features=init_features)

# Load the state dictionary into the model instance
model.load_state_dict(torch.load(model_path))

# Move the model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.to(device)
else:
    device = torch.device("cpu")

# Convert the model to float if needed
model = model.float()

# load a test dataset (SB 22 and 23)

data_directory = DATA_DIR + os.sep + 'BFE_A_05_data_snippets' + os.sep  # replace it with your own directory
# data_directory = 'DATA_DIR' + os.sep + 'SB23_15_SB31_08_handmade_labels_06_10' + os.sep


file_IDs = []

for filename in os.listdir(data_directory):
    if filename.endswith(".pt"):  # Assuming the files have a .pt extension
        file_id = int(filename.split("_")[3])  # Extract the ID number from the filename
        # Check if the ID is within the specified ranges for boreholes ST1 and MB8
        if (1 <= file_id <= 76):
            file_IDs.append(os.path.join(data_directory, filename))


complete_dataset = Dataset_np(file_IDs)

selected_batch_size = len(complete_dataset)  # Use the entire dataset for evaluation

# Create a DataLoader using the flipped dataset
test_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=selected_batch_size, shuffle=False)
# now for every Id from file_Ids, we will search into the metadata, to define which borehole it is. THen, we will extract the borehole diameter correspoding, form the  file_information
path_to_metadata_files = DATA_DIR + 'BFE_A_05_input_HS' + os.sep
metadata_files = pd.read_csv(path_to_metadata_files + 'BFE_A_05_metadata.csv', sep=',')

image_segmentation_pairs = []
all_images = []
all_segmentations = []
all_ids = []
all_depths = []

threshold = 0.70

output_dir = 'BFE_05_final_model_geometrical_extraction'  # Directory to save the test results
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for images, image_id in test_loader:
        images = images.to(device).float()
        outputs = model(images)
        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(1)

        # Save the predicted probabilities
        predicted_probs = outputs.cpu().numpy()

        predicted_segmentation = (outputs >= threshold).float()

        # Save the predicted segmentation masks
        all_images.append(images)
        all_segmentations.append(predicted_segmentation)
        #convert image_id so split can be used
        #convert id to real number
        for image_path in image_id:
        # Extract the ID number from the file path by splitting the string and getting the 4th element
            id = int(image_path.split("_")[7])
            # Extract the image ID from the tuple
            # search for image IDs in the metadata file, assign the 'Start Depth(m)' and End Depth(m) to the corresponding image ID, then create a vector with a value of depth for each pixel in the image (allimages have 360 pixels height)
            start_depth = metadata_files.loc[metadata_files['id'] == int(id), 'Start Depth (m)'].values[0]
            end_depth = metadata_files.loc[metadata_files['id'] == int(id), 'End Depth (m)'].values[0]
            depth_vector = np.linspace(start_depth, end_depth, 360)
            depth_vector = np.expand_dims(depth_vector, axis=1)
            all_depths.append(torch.tensor([depth_vector]))
            all_ids.append(torch.tensor([id]))

# put separately all images, all predicted_mask and all image ID on top of each other, with torch.cat
images = torch.cat(all_images, dim=0)
predicted_segmentations = torch.cat(all_segmentations, dim=0)
#convert_into_numpy
images = images.cpu().detach().numpy()
predicted_segmentations = predicted_segmentations.cpu().detach().numpy()


ids = torch.cat(all_ids, dim=0)
depths = torch.cat(all_depths, dim=0)
ids = ids.cpu().detach().numpy()
depths = depths.cpu().detach().numpy()
concatenated_image = np.concatenate(images, axis=0)

# Concatenate segmentations along the first axis (rows)
concatenated_segmentations = np.concatenate(predicted_segmentations, axis=0)

# Similarly, concatenate other arrays (ids and depths) if needed
concatenated_ids = ids
concatenated_depths = np.concatenate(depths, axis=0)

image_segmentation_pairs.extend(zip(images, predicted_segmentations, ids, depths))

# order the image_segmentation_pairs by the image ID
image_segmentation_pairs = sorted(image_segmentation_pairs, key=lambda x: x[2])



original_image_shape = images.shape
# Convert masks to uint8 if necessary


# -------------- parameters fof the LAS file and the borehole ------------------- #

vertical_resolution = 0.00417# [m/pixel] for all SB boreholes
diameter = 0.096  # [m] for all SB boreholes
perimeter = np.pi * diameter  # [m]
horizontal_resolution = perimeter / 360  # Since 360 pixels represent the full perimeter in m
# -------------------------------------------------------- #

total_masks_with_sin_fit = []
valid_masks = []
all_dips = []
all_azimuths = []
all_percentages = []
all_geometrical_parameters = []
all_sin_depths = []
dip_azimuth_borehole_all = []
# Loop through the image_segmentation_pairs
# extract the contours of predicted masks


masks, list_x_values, list_y_values = separate_masks(concatenated_segmentations, concatenated_image.shape, concatenated_depths)



for ids, mask in enumerate(masks):
    np_list_x_values = np.array(list_x_values[ids])
    np_list_y_values = np.array(list_y_values[ids])

    # Check if there is a mask and if it's wide enough to be considered as a fracture
    if np_list_x_values.size > 0 and np_list_y_values.size > 0:
        x_values = np_list_x_values
        y_values = np_list_y_values

        min_x_values = np.min(x_values)
        max_x_values = np.max(x_values)
        range_x_values = max_x_values - min_x_values
        x_upper, y_upper, x_lower, y_lower, apertures = [], [], [], [], []
        if range_x_values >= 100:
            try:
                params, _ = curve_fit(sinusoidal, x_values, y_values, maxfev=10000)
            except RuntimeError as e:
                print(f"RuntimeError: {e}. Skipping to the next mask.")
                continue
        else:
            print("Not enough data points for curve fitting. Restarting the entire loop.")
            continue  # Skip to the next mask if there are not enough data points

        # Store the mask if there is a valid sinusoidal fit
        total_masks_with_sin_fit.append(mask)

        # Extract parameters from the fitting sinusoidal curve
        A, B, C, D = params
        all_sin_depths.append(D)
        # Calculate high-side azimuth and dip
        # Calculate high-side azimuth and dip
        high_side_azimuth = ((C)*(360/(2*np.pi)))
        if high_side_azimuth < 0:
            high_side_azimuth += math.ceil(abs( high_side_azimuth) / 360) * 360
        elif  high_side_azimuth >= 360:
            high_side_azimuth -= math.floor(high_side_azimuth / 360) * 360

        all_azimuths.append(high_side_azimuth)
        high_side_dip = (90 - np.degrees(np.arctan((np.deg2rad(90) * horizontal_resolution) / (np.abs(A) * vertical_resolution))))
        alpha_angle = np.degrees(np.arctan((np.deg2rad(90) * horizontal_resolution) / (np.abs(A * vertical_resolution))))


        all_dips.append(high_side_dip)

        if high_side_dip:

            dip_azimuth_borehole = [high_side_azimuth, high_side_dip, D]
            dip_azimuth_borehole_all.append(dip_azimuth_borehole)

        else:
            print("The curve is not valid.")
        # Continue to the next mask if the curve is not valid

        # Create sinusoidal curve data points for the aperture extraction:
        num_x_sin = len(x_values)
        x_sin = np.linspace(min(x_values), max(x_values), num_x_sin)
        y_sin = sinusoidal(x_sin, A, B, C, D)
        x_sin_along_all_axis = np.linspace(0, 359, 360)
        y_sin_along_all_axis = sinusoidal(x_sin_along_all_axis, A, B, C, D)


        #define maximal distance for tangential projection

        max_distance = (1 / 2) * np.mean([
            np.sqrt((x_values[i] - x_sin[j]) ** 2 +
                    (y_values[i] - y_sin[j]) ** 2)
            for i in range(len(x_values)) for j in range(len(x_sin))
        ])



        # Define the parameters for the tangent calculation

        distances = []
        upper_corrected_apertures = []
        lower_corrected_apertures = []
        upper_apertures = []
        lower_apertures = []
        min_apertures = []
        max_apertures = []
        mean_apertures = []
        std_apertures = []
        all_apertures_mm = []
        upper_tangent_information = []
        lower_tangent_information = []
        upper_intersections = []
        lower_intersections = []
        upper_intersection_distances = []
        lower_intersection_distances = []
        curvatures = []
        x_tangents = []
        correction_factors = []
        over_estimation_factors = []

        # calculate the beta angle for each pixel along the azimuthal axis
        beta_angle_vector = calculate_beta_angle_vector(x_sin, x_values, x_sin_along_all_axis, y_sin_along_all_axis,
                                                        D)
        # define dx as the x distance between two points on the sinusoidal curve to calculate the tangent
        for i in range(len(x_sin)):
            # calculate dx adjusted and apply the factor
            dx = 0.05
            factor = 60

            #return the coordinates of tangents
            slope, y_tangent, x_tangent, curvature = calculate_tangent_sinus(x_sin[i], A, B, C, D, dx)
            perp_slope = calculate_perpendicular_slope(slope)


            if len(x_sin) == len(beta_angle_vector):
                beta_angle = beta_angle_vector[i]
            else:
                # repeat the last value of beta angle to have the same length as x_sin
                beta_angle = beta_angle_vector[-1]

            #calculate the correction factor

            correction_factor = ((np.sin(np.deg2rad(alpha_angle)) * np.cos(np.deg2rad(beta_angle))) ** 2 + (np.sin(np.deg2rad(alpha_angle)) * np.sin(np.deg2rad(beta_angle))) ** 2 + (np.cos(np.deg2rad(alpha_angle)) * np.cos(np.deg2rad(beta_angle))) ** 2) ** (0.5)
            over_estimation_factor = 1 / correction_factor

            # find the closest points on the data to the tangent line for upper and lower boundaries

            upper_closest_point, lower_closest_point = find_closest_points(x_values, y_values, x_tangent, y_tangent,
                                                                           perp_slope)
            x_tangents.append(x_tangent)

            curvatures.append(curvature)
            correction_factors.append(correction_factor)
            over_estimation_factors.append(over_estimation_factor)

            if upper_closest_point:
                upper_tangent_distance = np.sqrt(
                    ((upper_closest_point[0] - x_sin[i])) ** 2 + (
                        (upper_closest_point[1] - y_sin[i])) ** 2)

                if upper_tangent_distance <= max_distance:
                    upper_intersections.append(upper_closest_point)
                    # plot_results(x_sin[:i + 1], y_sin[:i + 1], x_values[:i + 1], y_values[:i + 1], intersections)

                    # Calculate the distance between the intersection point and the point on the sinusoidal curve
                    intersection_distance_x = np.abs(upper_closest_point[0] - x_sin[i])
                    intersection_distance_y = np.abs(upper_closest_point[1] - y_sin[i])
                    upper_intersection_distance_m = np.sqrt(
                        (intersection_distance_x * horizontal_resolution) ** 2 + (
                                intersection_distance_y * vertical_resolution) ** 2)

                    upper_intersection_m_corrected = upper_intersection_distance_m * correction_factor
                    upper_intersection_mm_corrected = upper_intersection_m_corrected * 1000
                    upper_intersection_distances.append(upper_intersection_mm_corrected)
                    # store the duo intersection points and corresponding sinsuoidal values
                    upper_tangent_information.append((upper_closest_point, (x_sin[i], y_sin[i])))

            if lower_closest_point:
                lower_tangent_distance = np.sqrt(
                    ((lower_closest_point[0] - x_sin[i])) ** 2 + (
                        (lower_closest_point[1] - y_sin[i])) ** 2)

                if lower_tangent_distance <= max_distance:
                    lower_intersections.append(lower_closest_point)
                    # plot_results(x_sin[:i + 1], y_sin[:i + 1], x_values[:i + 1], y_values[:i + 1], intersections)

                    # Calculate the distance between the intersection point and the point on the sinusoidal curve
                    intersection_distance_x = np.abs(lower_closest_point[0] - x_sin[i])
                    intersection_distance_y = np.abs(lower_closest_point[1] - y_sin[i])
                    lower_intersection_distance_m = np.sqrt(
                        (intersection_distance_x * horizontal_resolution) ** 2 + (
                                intersection_distance_y * vertical_resolution) ** 2)
                    lower_intersection_m_corrected = lower_intersection_distance_m * correction_factor
                    lower_intersection_mm_corrected = lower_intersection_m_corrected * 1000

                    lower_intersection_distances.append(lower_intersection_mm_corrected)
                    # store the duo intersection points and corresponding sinusoidal values
                    lower_tangent_information.append((lower_closest_point, (x_sin[i], y_sin[i])))

                    # convert to numpy array
        upper_aperture_mm_corrected = np.array(upper_intersection_distances)
        lower_aperture_mm_corrected = np.array(lower_intersection_distances)

        #convert resolutions in mm, to compare them with the extracted apertures. Take out the apertures which are smaller.
        horizontal_resolution_mm = horizontal_resolution * 1000
        vertical_resolution_mm = vertical_resolution * 1000
        # choose the smallest resolution between the vertical and horizontal resolution
        min_resolution_mm = np.min([horizontal_resolution_mm, vertical_resolution_mm])

        # within lower and upper apertures, only take the values which are > min_resolution_mm
        upper_aperture_mm_corrected = upper_aperture_mm_corrected[upper_aperture_mm_corrected > min_resolution_mm]
        lower_aperture_mm_corrected = lower_aperture_mm_corrected[lower_aperture_mm_corrected > min_resolution_mm]

        #take all apertures for one single segmentation and calculate statistics

        all_apertures_mm = pd.concat([pd.Series(upper_aperture_mm_corrected), pd.Series(lower_aperture_mm_corrected)])
        min_aperture_mm = np.min(all_apertures_mm)
        max_aperture_mm = np.max(all_apertures_mm)
        mean_aperture_mm = np.mean(all_apertures_mm)
        print(f"Mean aperture: {mean_aperture_mm:.2f} mm")
        std_aperture_mm = np.std(all_apertures_mm)
        print(f"Standard deviation of aperture: {std_aperture_mm:.2f} mm")
        mean_curvature = np.mean(curvatures)

        #store all geometrical parameters, so they can be savec in a .csv file
        if mean_aperture_mm > min_resolution_mm:
        # store the min, max, mean and std of the apertures
            geometrical_parameters = [high_side_azimuth, high_side_dip, alpha_angle, D, mean_aperture_mm, std_aperture_mm]
            all_geometrical_parameters.append(geometrical_parameters)
        else:
            geometrical_parameters = [high_side_azimuth, high_side_dip, alpha_angle, D, 'aperture is smaller than the minimum pixel resolution', 'aperture is smaller than the minimum pixel resolution']
            all_geometrical_parameters.append(geometrical_parameters)

all_picked_structures_files = pd.read_csv(path_to_metadata_files + 'BFE_A_05_picked_structures.csv', skiprows=[1])

# Access the columns
picked_depth = all_picked_structures_files['Feature Depth']
picked_dip = all_picked_structures_files['Dip']
picked_azimuth = all_picked_structures_files['Azimuth']



atv_visible_structures_files = pd.read_csv(path_to_metadata_files + 'BFE_A_05_composite_for_structural_comparison.csv', skiprows=[1])

# Access the columns
atv_visible_structures_depth = atv_visible_structures_files['Feature Depth']
atv_visible_structures_dip = atv_visible_structures_files['Dip']
atv_visible_structures_azimuth = atv_visible_structures_files['Azimuth']

# Filter only the structures labeled as fractures
picked_fractures = atv_visible_structures_files[atv_visible_structures_files['Type'] == 1]

picked_fractures_depth = picked_fractures['Feature Depth']
picked_fractures_dip = picked_fractures['Dip']
picked_fractures_azimuth = picked_fractures['Azimuth']


fig = plt.figure(figsize=(9, 8))

# Create the stereonet plot
ax = fig.add_subplot(111, projection='stereonet')



fig = plt.figure(figsize=(9, 8))

# Plot poles
ax.pole(picked_azimuth, picked_dip, 'bo')
for all_azimuths, all_dips, all_sin_depths in dip_azimuth_borehole_all:
    ax.pole(all_azimuths, all_dips, 'ro')

# Plot poles
ax.pole(all_azimuths, all_dips)

# Color-coded density contour
cax = ax.density_contourf(all_azimuths, all_dips, measurement='poles', cmap='rainbow')

# Add a colorbar with a larger size
cbar = plt.colorbar(cax, orientation='vertical', label='Density of poles', pad=0.1, aspect=40)

# Define ticks and labels for the azimuth of the stereonet
azimuths = [0,  90,  180,  270]
labels = ['0°', '90°',  '180°',  '270°']

# Manually add azimuth labels with offset
scaling_factor = 2.0  # Adjust this factor to increase the label radius
for azimuth, label in zip(azimuths, labels):
    angle_rad = np.deg2rad(azimuth)
    x = scaling_factor * np.sin(angle_rad)
    y = scaling_factor * np.cos(angle_rad)
    ax.text(x, y, label, ha='center', va='center', fontsize=12)
ax.set_azimuth_ticks([])

# Add borehole name annotations
for dip, azimuth, borehole_name in dip_azimuth_borehole_all:
    x, y = mplstereonet.pole(azimuth, dip)  # Convert dip and azimuth to stereonet coordinates
    ax.text(x-0.03, y, borehole_name, ha='right', va='top', fontsize=8)

#add a grid
ax.grid(True)
plt.show()
file_path = os.path.join(output_dir, 'Stereonet_density_with_Cbar.png')
if os.path.exists(file_path):
    os.remove(file_path)
fig.savefig(file_path, dpi=200)
plt.close()

#scatter plot # Depth vs Dip and Azimuth


#create a figures with 2 subplots (1 line 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
for all_azimuths, all_dips, all_sin_depths in dip_azimuth_borehole_all:
    # Left y-axis for Dip
    ax1.scatter(atv_visible_structures_dip, atv_visible_structures_depth,  color='g', marker = 's', label='ATV visible structures (WellCAD)')
    ax1.scatter( picked_fractures_dip, picked_fractures_depth,color='r', marker = '^', label='labeled fractures (WellCAD)')
    ax1.scatter( all_dips, all_sin_depths,color='black',marker = 'x', label='segmented')

    ax1.set_ylabel('Depth [m]')
    ax1.set_xlabel('Dip [°]')
    ax1.tick_params(axis='y')

    # Right y-axis for Azimuth
    ax2.scatter( atv_visible_structures_azimuth, atv_visible_structures_depth, color='g',marker = 's', label='ATV visible structures (WellCAD)')
    ax2.scatter( picked_fractures_azimuth, picked_fractures_depth,color='r', marker = '^', label='labeled fractures (WellCAD)')
    ax2.scatter( all_azimuths, all_sin_depths, color='black', marker = 'x', label='segmented')

    ax2.set_xlabel('Azimuth [°]')
    ax2.set_ylabel('')
    #take out labels fro the ticks
    ax2.set_yticklabels([])
    ax2.tick_params(axis='y')

    #set the x_lim to be in the range of all sin depths
    ax1.set_ylim([np.min(concatenated_depths), np.max(concatenated_depths)])
    ax2.set_ylim([np.min(concatenated_depths), np.max(concatenated_depths)])

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    #show legend on the ax 2, loc upper right

plt.legend(['picks visible on ATV*', 'picks labeled as fractures*', 'segmented structures'],
           loc='upper right', bbox_to_anchor=(1.0, 1.150))  # Adjust the coordinates as needed#add the legend for both plots outside the plot

# Add a title to the whole figure
fig.suptitle('Dip and Azimuth as a Function of Depth, Borehole BFE 05')
plt.show()
file_path = os.path.join(output_dir, 'Plot_picked_dip_az_vs_segmentations_4_at_scale.png')
if os.path.exists(file_path):
    os.remove(file_path)
fig.savefig(file_path, dpi=2300)
plt.close()

#save a .csv file with all high_side dip, azimuth and depth
# Create a DataFrame with the results, ust pass 2-d input. shape=(28, 3, 28), only keep the first two dimensions
# Assume dip_azimuth_borehole_all is a list of lists or similar nested structure
# Convert it to a NumPy array
#  Assume dip_azimuth_borehole_all is a list of lists or tuples



array_all_geometrical_parameters = np.array(all_geometrical_parameters, dtype=object)

dip_azimuth_borehole_all_df = pd.DataFrame(array_all_geometrical_parameters, columns=['Azimuth [°]', 'Dip [°]','Alpha Angle [°]' , 'Depth [m]', 'Mean Aperture [mm]', 'Std Aperture [mm]'])

# Save the DataFrame to a CSV file
dip_azimuth_borehole_all_df.to_csv(os.path.join(output_dir, f'extraction_geometrical_information_structures_borehole_{borehole_name}_threshold_07.csv'), index=False)



































