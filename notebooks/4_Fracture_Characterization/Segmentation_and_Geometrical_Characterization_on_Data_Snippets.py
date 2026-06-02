
import os
import torch
from deeplogger.model_architectures_ATV import UNetOTV
import matplotlib.pyplot as plt
import torch.utils.data as data
from deeplogger import DATA_DIR
from torch.nn import BCELoss
from matplotlib.colors import ListedColormap
from skimage.transform import hough_line, hough_line_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import precision_score, recall_score, adjusted_rand_score, mutual_info_score, precision_recall_curve, auc
import seaborn as sns

import pickle
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
import mplstereonet
import matplotlib.patches as mpatches
from matplotlib.table import Table
import math


# here we define another dataset class, which will be used to load the data, but also the ids
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


def separate_masks(segmentation_mask, original_image_shape):
    # Ensure the segmentation mask is a binary image of type uint8
    if segmentation_mask.dtype != np.uint8:
        segmentation_mask = segmentation_mask.astype(np.uint8)

    if len(segmentation_mask.shape) > 2:
        raise ValueError("segmentation_mask should be a 2D array (grayscale image).")

    # If the mask is not binary, apply a threshold to convert it to binary
    _, segmentation_mask = cv2.threshold(segmentation_mask, 127, 255, cv2.THRESH_BINARY)

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
        mask_x_values_list.append(x_values)
        mask_y_values_list.append(y_values)

    return individual_masks, mask_x_values_list, mask_y_values_list


#for sinusoidal curve fit
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

def calculate_derivative(x, A, B, C):
    # Assuming A, B, C, D are coefficients of the sinusoidal function calculate the derivative of the function at x, return the slpope
    # Calculate the derivative of the sinusoidal function, but input the next x_value so that the tangent follows the curve
    slope = A * B * np.cos(B * x + C)

    return slope

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



###-------------------------------------------------------------------------------------------------------------###


# Load the model saved model
model_name = '2D_unet_model07_16with_almost_allATV_handmade_labels75_epochs'
model_file = '2D_unet_model07_16with_almost_allATV_handmade_labels75_epochs.pt'
model_path = '/home/pperritaz/git/deeplogger/output/Bedretto_models' + os.sep + model_file


in_channels = 1  # Number of input channels (RGB image)
out_channels = 1  # Number of output channels
init_features = 32  # Initial number of features

# Instantiate the model
model = UNetOTV(in_channels=in_channels, out_channels=out_channels, init_features=init_features)

# Move the model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.to(device)
else:
    device = torch.device("cpu")

# Load the state dictionary into the model instance
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# Convert the model to float if needed
model = model.float()

#------------------------------------- For SB Boreholes ------------------------------------#
# start evaluating the model
#load a test dataset (here as we don't have the data, we will use the same training data but flipped horizontally)
#
# data_directory = DATA_DIR + os.sep + 'SB23_15_SB31_08_handmade_labels_06_10' + os.sep  #replace it with your own directory
# # data_directory = 'DATA_DIR' + os.sep + 'SB23_15_SB31_08_handmade_labels_06_10' + os.sep
# output_base_dir = '/home/pperritaz/git/deeplogger/output/Bedretto_models' + os.sep
#
# # Ensure output directory exists
# os.makedirs(output_base_dir, exist_ok=True)
#
# # Load the dataset
# file_IDs = [os.path.join(data_directory, filename) for filename in os.listdir(data_directory)]
# complete_dataset = Dataset_np(file_IDs)
# complete_dataset = [(image, torch.where(indices == 1, 0, torch.where(indices == 2, 1, torch.where(indices == 3, 1, indices))), image_id) for image, indices, image_id in complete_dataset]
# batch_size = len(complete_dataset)
# num_images = len(complete_dataset)
# test_loader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=False)
# batch_size = len(test_loader)


#-------------------------------------------For 100 random images from Bedretto boreholes--------------------------------------------------#

# # load the configuration file to extract the paths from the test set
# file_name = '2D_unet_model07_16BCEloss_all_ATV_adam_batch_size_20_config.p'
# path_to_file = '/home/pperritaz/git/deeplogger/output/Bedretto_models'
# file_path = path_to_file + os.sep + file_name
#
# # Load the configuration file
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#
#
# test_ids = data["test_set_IDs"]
# file_paths = list(test_ids)
# complete_dataset = Dataset_np(file_paths)
# selected_batch_size = len(complete_dataset)  # Use the entire dataset for evaluation
#
# # Replace 1 with 0 and 2 with 1 in the dataset
# complete_dataset = [(image, torch.where(indices == 1, 0, torch.where(indices == 2, 1, torch.where(indices == 3, 1, indices))), image_id) for image, indices, image_id in complete_dataset]
#
# # Create a DataLoader using the flipped dataset
# test_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=selected_batch_size, shuffle=False)
# # now for every Id from file_Ids, we will search into the metadata, to define which borehole it is. THen, we will extract the borehole diameter correspoding, form the  file_information
# #
#
# path_to_metadata_files = DATA_DIR + 'Bedretto_Input_HS' + os.sep
# metadata_files = pd.read_csv(path_to_metadata_files + 'Bedretto_metadata.csv', sep=',')
# borehole_information = pd.read_excel(path_to_metadata_files + 'file_informations.xlsx')



#------------------------------------------------------ For BFE 05 borehole ---------------------------------------#


data_directory = DATA_DIR + os.sep + 'BFE_A_05_data_snippets' + os.sep  # replace it with your own directory
# data_directory = 'DATA_DIR' + os.sep + 'SB23_15_SB31_08_handmade_labels_06_10' + os.sep


file_IDs = []

for filename in os.listdir(data_directory):
    if filename.endswith(".pt"):  # Assuming the files have a .pt extension
        file_id = int(filename.split("_")[3])  # Extract the ID number from the filename
        # Check if the ID is within the specified ranges for boreholes ST1 and MB8
        if (1 <= file_id <= 75):
            file_IDs.append(os.path.join(data_directory, filename))


complete_dataset = Dataset_np(file_IDs)

selected_batch_size = len(complete_dataset)  # Use the entire dataset for evaluation

# Create a DataLoader using the flipped dataset
test_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=selected_batch_size, shuffle=False)
# now for every Id from file_Ids, we will search into the metadata, to define which borehole it is. THen, we will extract the borehole diameter correspoding, form the  file_information
path_to_metadata_files = DATA_DIR + 'BFE_A_05_input_HS' + os.sep
metadata_files = pd.read_csv(path_to_metadata_files + 'BFE_A_05_metadata.csv', sep=',')
borehole_name = 'BFE_05'


image_segmentation_pairs = []
all_images = []
all_segmentations = []
ids = []
all_ids = []
all_depths = []

threshold = 0.70

output_dir = 'BFE_05_final_model_geometrical_extraction'  # Directory to save the test results
os.makedirs(output_dir, exist_ok=True)


#make predictions with the model :

with torch.no_grad():
    for images, image_ids in test_loader:
        images = images.to(device).float()
        outputs = model(images)

        if len(outputs.shape) == 2:
            outputs = outputs.unsqueeze(1)

        # Save the predicted probabilities
        predicted_probs = outputs.cpu().numpy()

        predicted_segmentation = (outputs >= threshold).float()

        # Iterate over the images and image IDs in the batch
        for idx, image_id in enumerate(image_ids):
            # Extract the individual image and predicted segmentation for the current index
            single_image = images[idx].cpu().numpy()
            single_predicted_segmentation = predicted_segmentation[idx].cpu().numpy()

            # Extract the ID number from the file path by splitting the string
            id_number = int(image_id.split('_')[-2])

            # Find the corresponding start and end depth from the metadata file
            start_depth = metadata_files.loc[metadata_files['id'] == id_number, 'Start Depth (m)'].values[0]
            end_depth = metadata_files.loc[metadata_files['id'] == id_number, 'End Depth (m)'].values[0]

            # Create a depth vector with values for each pixel (assuming 360 pixels in height)
            depth_vector = np.linspace(start_depth, end_depth, 360)
            depth_tensor = torch.tensor(depth_vector).unsqueeze(1)

            # Store the individual results
            all_images.append(single_image)
            all_segmentations.append(single_predicted_segmentation)
            all_ids.append(id_number)
            all_depths.append(depth_tensor)

            # Pair the images, segmentations, ids, and depths
            image_segmentation_pairs.append((single_image, single_predicted_segmentation, id_number, depth_tensor))




      # Initialize lists to store the masks with a valid sinusoidal fit and the valid masks
total_masks_with_sin_fit = []
valid_masks = []
all_dips = []
all_azimuths = []
all_percentages = []
dip_azimuth_borehole_all = []

#plot the generated segemntations

for i, (image, predicted_mask, image_id, depth) in enumerate(image_segmentation_pairs):
    # Find the borehole name for the current image ID

    # -------------- parameters for the LAS file and the borehole ------------------- #
    vertical_resolution = 0.00417  # [m/pixel] for all SB boreholes
    diameter = 0.096  # [m]
    perimeter = np.pi * diameter  # [m]
    horizontal_resolution = perimeter / 360  # Since 360 pixels represent the full perimeter in m
    # -------------------------------------------------------- #

    # Convert depth to numpy array
    depth = depth.numpy()
    min_depth = np.min(depth)
    max_depth = np.max(depth)

    # Check the shape of the image tensor and adjust accordingly
    single_image = image  # Removes single-dimensional entries from the shape

    # If the image still has 3 dimensions, pick one (assuming it's grayscale or you need to pick a channel)
    if single_image.ndim == 3:
        single_image = single_image[0]  # Selecting the first channel or image in the batch

    single_predicted_mask = predicted_mask

    # Plot the image and the mask
    cmap1 = ListedColormap(['none', 'cornflowerblue'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].imshow(single_image, cmap='YlOrBr')
    axes[0].set_xticks([])
    axes[0].set_yticks([0, 359])
    axes[0].set_yticklabels([f"{min_depth:.2f}", f"{max_depth:.2f}"])
    axes[0].set_ylabel('Depth [m]')

    axes[1].imshow(single_image, cmap='YlOrBr')
    axes[1].imshow(single_predicted_mask, cmap=cmap1, label='Segmentation')
    mask_patch = mpatches.Patch(color='cornflowerblue', label='Segmentation')
    axes[1].legend(handles=[mask_patch], loc='upper right')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Add title and save the figure
    plt.suptitle(f"Image and Segmentation ID: {image_id} \n Classification Threshold : {threshold}", fontsize=15)
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'Image_segmentation_{image_id}_threshold_{threshold}.png')

    if os.path.exists(file_path):
        os.remove(file_path)

    plt.savefig(file_path, dpi=200)
    plt.close()  # Close the figure to free up memory

original_image_shape = single_image.shape
masks, list_x_values, list_y_values = separate_masks(single_predicted_mask, original_image_shape)


#extract the geometrical information per image snippet and plot them along

for ids, mask in enumerate(masks):
    np_list_x_values = np.array(list_x_values[ids])
    np_list_y_values = np.array(list_y_values[ids])

    # Check if there is a mask and if it's wide enough to be considered as a fracture
    if np_list_x_values.size > 0 and np_list_y_values.size > 0:
        y_mask = (np_list_y_values >= 1) & (np_list_y_values <= 358)
        x_values = np_list_x_values[y_mask]
        y_values = np_list_y_values[y_mask]

        min_x_values = np.min(x_values)
        max_x_values = np.max(x_values)
        range_x_values = max_x_values - min_x_values
        x_upper, y_upper, x_lower, y_lower, apertures = [], [], [], [], []
        #here, we select that the x_values should be at least 100 pixels of azimuthal range
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

        # Calculate high-side azimuth and dip
        high_side_azimuth = ((C) * (360 / (2 * np.pi)))
        if high_side_azimuth < 0:
            high_side_azimuth += math.ceil(abs(high_side_azimuth) / 360) * 360
        elif high_side_azimuth >= 360:
            high_side_azimuth -= math.floor(high_side_azimuth / 360) * 360

        high_side_dip = 90 - np.degrees(np.arctan((np.deg2rad(90) * horizontal_resolution) / (np.abs(A * vertical_resolution))))
        alpha_angle = np.degrees(np.arctan((np.deg2rad(90) * horizontal_resolution) / (np.abs(A * vertical_resolution))))


        # Create sinusoidal curve data points
        num_x_sin = len(x_values)
        x_sin = np.linspace(min(x_values), max(x_values), num_x_sin)
        y_sin = sinusoidal(x_sin, A, B, C, D)
        x_sin_along_all_axis = np.linspace(0, 359, 360)
        y_sin_along_all_axis = sinusoidal(x_sin_along_all_axis, A, B, C, D)


        # Calculate the percentage of the curve inside the contour
        valid_points = 0
        total_points = len(x_sin)

        for i in range(total_points):
            x_point = x_sin[i]
            y_point = y_sin[i]
            if cv2.pointPolygonTest(np.array(list(zip(x_values, y_values))), (x_point, y_point), False) >= 1:
                valid_points += 1

        percentage_inside_contour = (valid_points / total_points) * 100
        print(f"Percentage of the curve inside the contour: {percentage_inside_contour:.2f}%")
        all_percentages.append(percentage_inside_contour)
        if high_side_dip:
            all_dips.append(high_side_dip)
            all_azimuths.append(high_side_azimuth)
            #combined the high side dip, azimuth and borehole_name in a list
            #if the borehole name is 'CB1' replace it by MB1

            dip_azimuth_borehole = [high_side_dip, high_side_azimuth, borehole_name]
            valid_masks.append(percentage_inside_contour)
            dip_azimuth_borehole_all.append(dip_azimuth_borehole)

            plt.figure()
            cmap1 = ListedColormap(['none', 'cornflowerblue'])
            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            axes[0].imshow(image, cmap='YlOrBr')
            #take out ticks and labels
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            axes[1].imshow(image, cmap='YlOrBr')
            axes[1].imshow(mask, cmap=cmap1, label='Segmentation')
            axes[1].plot(x_sin_along_all_axis, y_sin_along_all_axis, linestyle ='--', color = 'black', label='Sinusoidal fit')

            #create a proxy artist with two legends
            mask_patch = mpatches.Patch(color='cornflowerblue', label='Segmentation')
            sin_patch = mpatches.Patch(color='black', linestyle='--', label='Sinusoidal fit')

            #create a legend containing both the mask and the sinusoidal fit
            axes[1].legend(handles=[mask_patch, sin_patch], loc='upper right')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            # add title to the figure
            plt.suptitle(f"Image and Segmentation with Sin Fit", fontsize=15)
            # add a text below the figure in the middle :
            plt.figtext(0.5, 0.9,
                        f"High-side azimuth/dip: {high_side_azimuth:.2f}°/{high_side_dip:.2f}°",
                        ha="center", fontsize=11, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
            file_path = os.path.join(output_dir, f'sin_fit_dip_az_{high_side_dip}.png')
            plt.tight_layout()
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path, dpi=200)
            plt.close()  # Close the figure to free up memory

        else:
            print("The curve is not valid.")
      # Continue to the next mask if the curve is not valid


        max_distance =(1/2) * np.mean([
            np.sqrt((x_values[i] - x_sin[j])  ** 2 +
                   (y_values[i] - y_sin[j]) ** 2)
            for i in range(len(x_values)) for j in range(len(x_sin))
        ])

        upper_intersections = []
        lower_intersections = []
        upper_intersection_distances = []
        lower_intersection_distances = []

        # Calculate the distances along the tangents
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
        curvatures  = []
        x_tangents = []
        correction_factors = []
        over_estimation_factors = []

        beta_angle_vector = calculate_beta_angle_vector(x_sin,x_values, x_sin_along_all_axis, y_sin_along_all_axis, D)

        #define dx as the x distance between two points on the sinusoidal curve to calculate the tangent
        for i in range(len(x_sin)):
            #calculate dx as the distance between each x_sin point
            dx = 0.05
            factor = 60
            slope, y_tangent, x_tangent, curvature = calculate_tangent_sinus(x_sin[i], A, B, C, D, dx)
            perp_slope = calculate_perpendicular_slope(slope)
            if len(x_sin) == len(beta_angle_vector):
                beta_angle = beta_angle_vector[i]
            else:
            # repeat the last value of beta angle to have the same length as x_sin
                beta_angle = beta_angle_vector[-1]

            correction_factor = ((np.sin(np.deg2rad(alpha_angle))*np.cos(np.deg2rad(beta_angle)))**2 + (np.sin(np.deg2rad(alpha_angle))*np.sin(np.deg2rad(beta_angle)))**2 + (np.cos(np.deg2rad(alpha_angle))*np.cos(np.deg2rad(beta_angle)))**2 ) **(0.5)
            over_estimation_factor = 1/correction_factor
            upper_closest_point, lower_closest_point = find_closest_points(x_values, y_values, x_tangent, y_tangent, perp_slope)
            x_tangents.append(x_tangent)

            curvatures.append(curvature)
            correction_factors.append(correction_factor)
            over_estimation_factors.append(over_estimation_factor)

            if upper_closest_point:
                upper_tangent_distance = np.sqrt(
                    ((upper_closest_point[0] - x_sin[i]) ) ** 2 + (
                                (upper_closest_point[1] - y_sin[i]) ) ** 2)

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
                        #store the duo intersection points and corresponding sinsuoidal values
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



                    #convert to numpy array
        upper_aperture_mm_corrected = np.array(upper_intersection_distances)
        lower_aperture_mm_corrected = np.array(lower_intersection_distances)

        horizontal_resolution_mm = horizontal_resolution * 1000
        vertical_resolution_mm = vertical_resolution * 1000
        #choose the smallest resolution between the vertical and horizontal resolution
        min_resolution_mm = np.min([horizontal_resolution_mm, vertical_resolution_mm])

            # within lower and upper apertures, only take the values which are > min_resolution_mm
        upper_aperture_mm_corrected = upper_aperture_mm_corrected[upper_aperture_mm_corrected > min_resolution_mm]
        lower_aperture_mm_corrected = lower_aperture_mm_corrected[lower_aperture_mm_corrected > min_resolution_mm]

        all_apertures_mm = pd.concat([pd.Series(upper_aperture_mm_corrected), pd.Series(lower_aperture_mm_corrected)])
        min_aperture_mm = np.min(all_apertures_mm)
        max_aperture_mm = np.max(all_apertures_mm)
        mean_aperture_mm = np.mean(all_apertures_mm)
        std_aperture_mm = np.std(all_apertures_mm)
        mean_curvature = np.mean(curvatures)

    # Create a colormap
    cmap1 = ListedColormap(['none', 'cornflowerblue'])

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Plot the first subplot (top left)
    axes[0, 0].imshow(image, cmap='YlOrBr')
    axes[0, 0].set_title('Image')
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_aspect('equal')

     # Plot the second subplot (top right)
    axes[0, 1].imshow(image, cmap='YlOrBr')
    axes[0, 1].plot(x_values, y_values, color='cornflowerblue', alpha=0.7, label='Segmentation')
    axes[0, 1].set_title('Image with Segmentation')
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    axes[0, 1].legend()
    axes[0, 1].set_aspect('equal')

     # Plot the third subplot (bottom left)
    axes[1, 0].imshow(image, cmap='YlOrBr')
    axes[1, 0].set_title('Segmentation with fitted sinuosoidal')
    axes[1, 0].plot(x_values, y_values, color='cornflowerblue', label='Segmentation')
    axes[1, 0].plot(x_sin_along_all_axis, y_sin_along_all_axis, color='black', linestyle='--',
                     label='Fitted Sinusoidal')
     #plot the legend
    axes[1, 0].legend()
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    axes[1 , 0].set_aspect('equal')

    step_size = 1  # Adjust step size to plot every third point
    for i in range(0, len(upper_tangent_information), step_size):
        intersection, sinusoidal_point = upper_tangent_information[i]
        axes[1, 1].plot([sinusoidal_point[0], intersection[0]], [sinusoidal_point[1], intersection[1]],
                         color='red')

        axes[1, 1].scatter(intersection[0], intersection[1], color='black', marker='.')
        axes[1, 1].plot(x_sin, y_sin,
                           color='black',linestyle = '--')
         # axes[1, 1].scatter(sinusoidal_point[0], sinusoidal_point[1], color='black')
    for i in range(0, len(lower_tangent_information), step_size):
        intersection, sinusoidal_point = lower_tangent_information[i]
        axes[1, 1].plot([sinusoidal_point[0], intersection[0]], [sinusoidal_point[1], intersection[1]],
                         color='red')
        axes[1, 1].scatter(intersection[0], intersection[1], color='black', marker='.')
         # axes[1, 1].scatter(sinusoidal_point[0], sinusoidal_point[1], color='black')
         # invert the y-axis to have the same orientation as the borehole

    axes[1, 1].set_title(f'Tangent and intersection points')
    axes[1, 1].set_ylim(360,0)
    axes[1, 1].set_xlim(0, 360)
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xticks([])
    axes[1, 1].set_aspect('equal')
     #only add one legend for the intersection points with icons and the corresponding colors
    axes[1, 1].legend(['Projected Tangents', 'Intersection Points'], loc='upper right')


     # Write down the calculated D and JRC values for upper and lower profiles between the upper and lower right plots

    text = (f"Aperture (Mean ± std): {mean_aperture_mm:.2f} ± {std_aperture_mm:.2f} mm")
     #show the text below the [1,1] plot
    axes[1, 1].text(0.5, -0.1, text, ha='center', va='center', transform=axes[1, 1].transAxes,
                     bbox=dict(facecolor='orange', alpha=0.5))

     # Add a title to the entire figure
    plt.suptitle("Image, Segmentation and Projected Tangents for Aperture Extraction", fontsize=15)

     # Save the figure
    file_path = os.path.join(output_dir, f'plot_projected_apertures_{high_side_dip}.png')
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, dpi = 200)
    plt.close()  # Close the figure to free up memory

