
import os
import torch
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
import pandas as pd
import cv2
import sys


#here, we will  create a  dataset of images if 360x 360 dimensions witha  background of 0 and containing perfect sinusoidals (with a certain aperture, which can be defined), its location should contain 1s.


class SinusoidalDataset(data.Dataset):
    def __init__(self, length, aperture, image_size=360):
        self.length = length
        self.aperture = aperture
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.generate_images()

    def generate_images(self):
        for i in range(self.length):
            image = np.zeros((self.image_size, self.image_size))
            label = np.zeros((self.image_size, self.image_size))
            # Generate a random sinusoidal
            amplitude = np.random.uniform(0, self.image_size/4)
            frequency = 1
            phase_shift = np.random.uniform(0, 2 * np.pi)  # Use radians for phase shift
            x = np.linspace(0, 2 * math.pi, self.image_size)
            y = amplitude * np.sin(frequency * x + phase_shift) + self.image_size / 2
            y = np.clip(y, 0, self.image_size - 1)
            y = y.astype(int)
            for j in range(self.image_size):
                image[y[j] - self.aperture : y[j] + self.aperture, j] = 1
                label[y[j] - self.aperture : y[j] + self.aperture, j] = 1
            # Add an ID to the image (image_ids), so it can be called for evaluation
            image_id = f"image_{i}"
            self.images.append((torch.tensor(image), image_id))
            self.labels.append((torch.tensor(label), image_id))  # Include aperture in labels

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.images[idx][0], self.labels[idx][0], self.images[idx][1]


length = 5
defined_aperture = 10
# Define the dataset
dataset = SinusoidalDataset(length, defined_aperture)
# Define the data loader
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True)

# call  and load the saved model

model_name = '2D_unet_model05_13with_almost_allATV_handmade_labels300_epochs'
model_file = '2D_unet_model05_13with_almost_allATV_handmade_labels300_epochs.pt'
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



def evaluate_model(model, testloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_losses = []
    num_correct = 0
    total_samples = 0
    image_segmentation_pairs = []  # List to store pairs of images, segmentation masks, and image IDs

    with torch.no_grad():
        for data in testloader:
            images, indices, image_ids = data  # Unpack the tuple to get the image IDs
            images = images.to(device).float()  # Convert input to float
            indices = indices.unsqueeze(1).to(device).float()  # Add channel dimension and convert target to float

            # Forward pass
            outputs = model(images)

            # Ensure the output has the channel dimension if it's missing
            if len(outputs.shape) == 2:
                outputs = outputs.unsqueeze(1)

            # Make sure the target mask has the same dimensions as the output
            indices = indices.view_as(outputs)

            loss = criterion(outputs, indices)
            test_losses.append(loss.item())

            # Calculate accuracy
            predicted = torch.round(outputs)  # Assuming binary segmentation
            num_correct += (predicted == indices).sum().item()
            total_samples += indices.numel()

            # Convert predicted segmentation to match the format of indices (0 and 1)
            segmentation = predicted.byte()  # Convert to byte tensor (0 and 1)
            indices = indices.byte()  # Convert to byte tensor (0 and 1)
            # Extend the list with tuples of images, segmentation masks, and image IDs
            image_segmentation_pairs.extend(zip(images.cpu(), segmentation.cpu(), indices.cpu(), image_ids))

    return test_losses, num_correct, total_samples, image_segmentation_pairs



#necessary functions to extract and characterize generated segmentations



test_losses, num_correct, total_samples, image_segmentation_pairs = evaluate_model(model=model, testloader=data_loader, criterion=BCELoss(), device=device)


#calculate the fractal dimension of the fracture surface images

# Extract the segmentation mask of the fracture surface
#to retrieve contours, no chain approximation is used, to be sure that every point of the  contour is taken.
#Also, the contours are retrieved in a two-level hierarchy, to get the outer and inner contours and get the most points to recontruct the roughness profile of the fracture surface

def extract_contours(segmentation_mask):
    # Find contours in the binary segmentation mask
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return contours

def separate_masks(segmentation_mask, original_image_shape):
    # Convert the segmentation mask to CV_8UC1 format
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
        mask_x_values_list.append(x_values)
        mask_y_values_list.append(y_values)

    return individual_masks, mask_x_values_list, mask_y_values_list


#for sinusoidal curve fit
def sinusoidal(x, A, B, C, D):
    # Ensure that x is an array
    x = np.array(x)
    B = 2 * np.pi / 360
    return A * np.sin(B * x + C) + D


#loop and plots to characterize the generated fracture surfaces


output_dir = 'tests_perfect_cases'  # Directory to save the test results
os.makedirs(output_dir, exist_ok=True)

# Flag to restart the loop if an error occurs

for image, label,image_ids in data_loader:
    # Find the borehole name for the current image ID
    borehole_diameter = 0.1  # [m]
    vertical_resolution = 0.004  # [m/pixel]

    # Calculate the equivalent distance of 1 pixel
    perimeter = np.pi * borehole_diameter
    pixel_distance_horizontal = perimeter / 360  # Since 360 pixels represent the full perimeter in m

    horizontal_resolution = pixel_distance_horizontal  # [m/pixel]


    # convert the image and predicted mask into images readable with cv2

    image = image.squeeze().numpy()
    original_image_shape = image.shape

    # generate individual masks for each fractures
    # start with a box size of 1
    images, list_x_values, list_y_values = separate_masks(image, original_image_shape)

    for ids, image in enumerate(images):

        np_list_x_values = np.array(list_x_values[ids])
        np_list_y_values = np.array(list_y_values[ids])

        # Apply the fourier transform to the mask
        if np_list_x_values.size > 0 and np_list_y_values.size > 0:
            # Create a mask for y-values within the desired range
            y_image = (np_list_y_values >= 1) & (np_list_y_values <= 358)
            # Apply the mask to both x and y arrays
            x_values = np_list_x_values[y_image]
            y_values = np_list_y_values[y_image]

            # Calculate the roughness profile of the fracture surface

            x_upper, y_upper, x_lower, y_lower, apertures = [], [], [], [], []
            if len(x_values) >= 350:
                try:
                    params, _ = curve_fit(sinusoidal, x_values, y_values, maxfev=10000)
                except RuntimeError as e:
                    print(f"RuntimeError: {e}. Skipping to the next mask.")
                    continue
            else:
                print("Not enough data points for curve fitting. Restarting the entire loop.")
                restart_flag = True
                continue  # continue the inner loop if there are enough segmentation masks

            # extract parameters form the fitting sinusoidal curve to determine High Side dip and azimuth
            A, B, C, D = params  # only parameters A (Amplitude) and C (phase shift) are interesting for us

            # high-side azimuth
            high_side_azimuth = ((C) * (360 / (2 * np.pi)))
            # Adjust azimuth if it's negative or above 360

            if high_side_azimuth < 0 or high_side_azimuth > 0:
                if high_side_azimuth < 0:
                    high_side_azimuth += math.ceil(abs(high_side_azimuth) / 360) * 360
                elif high_side_azimuth >= 360:
                    high_side_azimuth -= math.floor(high_side_azimuth / 360) * 360

            high_side_dip = 90 - np.degrees(np.arctan((np.deg2rad(90) * horizontal_resolution) / (np.abs(A * vertical_resolution))))
            alpha_angle = np.degrees(np.arctan((np.deg2rad(90) * horizontal_resolution) / (np.abs(A * vertical_resolution))))


            # to project the tangents along the sinusoidal curve, we need to calculate the intersection points of the tangents with the curve
            # but first, we need to actually create the data points of the curve
            num_x_sin = len(x_values)
            x_sin = np.linspace(min(x_values), max(x_values), num_x_sin)
            y_sin = sinusoidal(x_sin, A, B, C, D)

            x_sin_along_all_axis = np.linspace(0, 359, 360)
            y_sin_along_all_axis = sinusoidal(x_sin_along_all_axis, A, B, C, D)



            # calculate the tangent at some point on the curve
            def calculate_tangent_sinus(x, A, B, C, D, dx):
                # Calculate y, y', and y''
                y = sinusoidal(x, A, B, C, D)
                y_prime = ((sinusoidal(x + dx , A, B, C, D) - sinusoidal(x - dx , A, B, C, D)) / dx)
                y_double_prime = ((sinusoidal(x + dx / 2, A, B, C, D) - 2 * y + sinusoidal(x - dx / 2, A, B, C, D)) / (
                        dx / 2) ** 2)

                # Calculate curvature
                curvature = np.abs(y_double_prime) / ((1 + y_prime ** 2) ** (3 / 2))

                perp_slope = -1/y_prime

                return perp_slope, y, x, curvature


            def calculate_derivative(x, A, B, C):
                # Assuming A, B, C, D are coefficients of the sinusoidal function calculate the derivative of the function at x, return the slpope
                # Calculate the derivative of the sinusoidal function, but input the next x_value so that the tangent follows the curve
                slope = A * B * np.cos(B * x + C)

                return slope


            def calculate_perpendicular_slope(x, A, B, C, D, dx):
                # Calculate the original slope (derivative) at three points
                y_prime1 = (sinusoidal(x - dx, A, B, C, D) - sinusoidal(x, A, B, C, D)) / dx
                curvature_1 = np.abs(y_prime1) / ((1 + y_prime1 ** 2) ** (3 / 2))
                y_prime2 = (sinusoidal(x, A, B, C, D) - sinusoidal(x + dx, A, B, C, D)) / dx
                curvature_2 = np.abs(y_prime2) / ((1 + y_prime2 ** 2) ** (3 / 2))


                # Calculate the average slope between the three points
                perpendicular_slope = -1 / ((y_prime1 + y_prime2) / 2)

                # Calculate the corresponding y value (tangent point)
                y_tangent = sinusoidal(x, A, B, C, D)

                # Calculate the curvature (optional, if needed)
                curvature = curvature_1

                # Return all relevant values as a tuple
                return perpendicular_slope, x, y_tangent, curvature

            # Function to find the closest point on the data to the tangent line
            # Function to find the closest point on the data to the tangent line

            def find_closest_points(x_values, y_values, x_tangent, y_tangent, perp_slope):
                upper_closest_point = None
                lower_closest_point = None
                min_left_distance = float('inf')
                min_right_distance = float('inf')
                tolerance = 0.5 # Define a tolerance for floating-point comparison

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
                beta_angle_vector_along_all_axis = np.arange(0 - crossing_index,
                                                             len(x_sin_along_all_axis) - crossing_index, 1)

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


            max_distance = (1/2) * np.mean([
                np.sqrt((x_values[i] - x_sin[j]) ** 2 +
                        (y_values[i] - y_sin[j]) ** 2)
                for i in range(len(x_values)) for j in range(len(x_sin))
            ])

            upper_intersections = []
            lower_intersections = []
            upper_intersection_distances = []
            lower_intersection_distances = []
            upper_tangent_distances = []
            lower_tangent_distances = []
            lower_tangent_distances_corrected = []
            upper_tangent_distances_corrected = []

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
            upper_tangent_information = []
            lower_tangent_information = []
            curvatures = []
            x_tangents = []
            correction_factors = []
            over_estimation_factors = []

            beta_angle_vector = calculate_beta_angle_vector(x_sin, x_values, x_sin_along_all_axis, y_sin_along_all_axis,
                                                            D)

            # define dx as the x distance between two points on the sinusoidal curve to calculate the tangent
            for i in range(len(x_sin)):
                # calculate dx as the distance between each x_sin point
                dx = 2
                # slope, y_tangent, x_tangent, curvature = calculate_tangent_sinus(x_sin[i], A, B, C, D, dx)
                # from 0.0050 to higher, the factor is tuned as function of the curvature. the higher the curvature, the lower the factor
                # tune the factor to get the best results
                current_x_sin = x_sin[i]
                current_y_sin = y_sin[i]
                perp_slope, x_tangent, y_tangent, curvature = calculate_tangent_sinus(current_x_sin, A, B, C, D, dx)
                beta_angle = beta_angle_vector[i]
                correction_factor = ((np.sin(np.deg2rad(alpha_angle)) * np.cos(np.deg2rad(beta_angle))) ** 2 + (
                            np.sin(np.deg2rad(alpha_angle)) * np.sin(np.deg2rad(beta_angle))) ** 2 + (
                                                 np.cos(np.deg2rad(alpha_angle)) * np.cos(
                                             np.deg2rad(beta_angle))) ** 2) ** (0.5)
                over_estimation_factor = 1 / correction_factor

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


                    if upper_tangent_distance <= max_distance :
                        upper_intersections.append(upper_closest_point)
                        # plot_results(x_sin[:i + 1], y_sin[:i + 1], x_values[:i + 1], y_values[:i + 1], intersections)

                        # Calculate the distance between the intersection point and the point on the sinusoidal curve
                        intersection_distance_x = np.abs(upper_closest_point[0] - x_sin[i])
                        intersection_distance_y = np.abs(upper_closest_point[1] - y_sin[i])


                        upper_intersection_distance_m = np.sqrt(
                            (intersection_distance_x * horizontal_resolution) ** 2 + (
                                    intersection_distance_y * vertical_resolution) ** 2)

                        upper_intersection_m_corrected = upper_intersection_distance_m * correction_factor
                        upper_tangent_distance_corrected = upper_tangent_distance * correction_factor
                        upper_intersection_mm_corrected = upper_intersection_m_corrected * 1000
                        upper_intersection_distances.append(upper_intersection_mm_corrected)
                        # store the duo intersection points and corresponding sinsuoidal values
                        upper_tangent_information.append((upper_closest_point, (x_sin[i], y_sin[i])))
                        upper_tangent_distances_corrected.append(upper_tangent_distance_corrected)
                        upper_tangent_distances.append(upper_tangent_distance)

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
                        lower_tangent_distance_corrected = lower_tangent_distance * correction_factor
                        lower_intersection_mm_corrected = lower_intersection_m_corrected * 1000

                        lower_intersection_distances.append(lower_intersection_mm_corrected)
                        # store the duo intersection points and corresponding sinusoidal values
                        lower_tangent_information.append((lower_closest_point, (x_sin[i], y_sin[i])))
                        lower_tangent_distances_corrected.append(lower_tangent_distance_corrected)
                        lower_tangent_distances.append(lower_tangent_distance)

                        # convert to numpy array
            upper_aperture_mm_corrected = np.array(upper_intersection_distances)
            lower_aperture_mm_corrected = np.array(lower_intersection_distances)

            horizontal_resolution_mm = horizontal_resolution * 1000
            vertical_resolution_mm = vertical_resolution * 1000
            # choose the smallest resolution between the vertical and horizontal resolution
            min_resolution_mm = np.min([horizontal_resolution_mm, vertical_resolution_mm])

            # within lower and upper apertures, only take the values which are > min_resolution_mm
            upper_aperture_mm_corrected = upper_aperture_mm_corrected[upper_aperture_mm_corrected > min_resolution_mm]
            lower_aperture_mm_corrected = lower_aperture_mm_corrected[lower_aperture_mm_corrected > min_resolution_mm]

            all_apertures_mm = pd.concat(
                [pd.Series(upper_aperture_mm_corrected), pd.Series(lower_aperture_mm_corrected)])
            min_aperture_mm = np.min(all_apertures_mm)
            max_aperture_mm = np.max(all_apertures_mm)
            mean_aperture_mm = np.mean(all_apertures_mm)
            std_aperture_mm = np.std(all_apertures_mm)
            mean_curvature = np.mean(curvatures)
            all_tangent_distances = np.concatenate([upper_tangent_distances, lower_tangent_distances])
            all_tangent_distances_corrected = np.concatenate([upper_tangent_distances_corrected, lower_tangent_distances_corrected])
            mean_tangent_distance = np.mean(all_tangent_distances)
            mean_tangent_distance_corrected = np.mean(all_tangent_distances_corrected)

            print(f"Mean tangent distance: {mean_tangent_distance:.2f} [pixel]",f"Corrected mean tangent distance: {mean_tangent_distance_corrected:.2f} [pixel]", f"mean curvature: {mean_curvature}", f"Minimum aperture: {min_aperture_mm:.2f} mm", f"Maximum aperture: {max_aperture_mm:.2f} mm",
                  f"Mean aperture: {mean_aperture_mm:.2f} mm", f"Standard deviation: {std_aperture_mm:.2f} mm")
            plt.figure()
            plt.hist(all_apertures_mm, bins=50)
            plt.xlabel('Aperture [mm]')
            plt.ylabel('Frequency')
            plt.title(
                f'Aperture distribution, \n Mean tangent distance: {mean_tangent_distance:.2f} [pixel]", \n min: {min_aperture_mm:.2f} mm, max: {max_aperture_mm:.2f} mm, mean: {mean_aperture_mm:.2f} mm, std: {std_aperture_mm:.2f} mm')
            plt.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, f'hist_aperture_dx_{dx}_structure{ids}_aperture_{defined_aperture}.png')
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path)
            plt.close()  # Close the figure to free up memory

            # Plot the curvature distribution along the sinusoidal curve
            plt.figure()
            plt.plot(x_tangents, curvatures, label='Curvature Along the Sinusoidal Curve')
            plt.plot(x_sin, y_sin, label='Sinusoidal curve')
            plt.xlabel('X [pixel]')
            plt.ylabel('Y [pixel]')
            plt.title(f'Curvature distribution along the fitted sinusoidal curve, mean curvature: {mean_curvature:.4f}')
            # put y-axis in a log-scale
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, f'plot_curvature_dx_{dx}_structure{ids}.png')
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path)
            plt.close()  # Close the figure to free up memory

            plt.figure()
            # Plot the sinusoidal curve and the intersection points, with their corresponding tangents
            plt.plot(x_sin, y_sin, label='Sinusoidal curve')
            plt.scatter(x_values, y_values, color='red', label='Fracture surface points')

            # we will only plots 1 of 3 tangents, to actually see them on the plot. otherwise there are too many and it is too dense.
            step_size = 3  # Adjust step size to plot every x point
            for i in range(0, len(upper_tangent_information), step_size):
                intersection, sinusoidal_point = upper_tangent_information[i]
                plt.scatter(sinusoidal_point[0], sinusoidal_point[1], color='blue')
                plt.plot([sinusoidal_point[0], intersection[0]], [sinusoidal_point[1], intersection[1]], color='green')
                plt.scatter(intersection[0], intersection[1], color='green',
                            label='intersection points (a)' if i == 0 else "")

                plt.ylim(0, 360)
                plt.gca().invert_yaxis()
                plt.xlim(0, 360)
            for i in range(0, len(lower_tangent_information), step_size):
                intersection, sinusoidal_point = lower_tangent_information[i]
                plt.plot([sinusoidal_point[0], intersection[0]], [sinusoidal_point[1], intersection[1]],
                         color='orange')
                plt.scatter(intersection[0], intersection[1], color='orange',
                            label='intersection points (b)' if i == 0 else "")
                plt.ylim(0, 360)
                plt.gca().invert_xaxis()
                plt.xlim(0, 360)
                # invert the y-axis to have the same orientation as the borehole
            plt.ylim(0, 360)

            plt.xlim(0, 360)
            plt.xlabel('X [pixel]')
            plt.ylabel('Y [pixel]')
            plt.title(
                f'Sinusoidal curve and intersection points, ID: {ids}, mean curvature: {mean_curvature:.4f}, dx: {dx:.2f} pixel \n mean aperture: {mean_aperture_mm:.2f} mm, std aperture: {std_aperture_mm:.2f} mm \n Mean tangent distance: {mean_tangent_distance:.2f} [pixel]')
            plt.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, f'sinusoidal_curve_intersections_dx_{dx}_aperture{i}.png')
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path)
            plt.close()  # Close the figure to free up memory
            # Plot the mask with the fitted sinusoidal curve


            cmap1 = ListedColormap(['none', 'red'])
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            num_images = len(image_segmentation_pairs)

            # print(y_lower_profile)
            axes[0, 0].imshow(image, cmap='viridis')
            axes[0, 0].set_title(f'Image {ids + 1}')

            axes[0, 1].imshow(image, cmap='viridis')
            axes[0, 1].imshow(label.squeeze(0), cmap=cmap1, alpha=0.7)
            axes[0, 1].set_title(f'Image with segmentation Mask')

            axes[1, 0].imshow(image, cmap='viridis')
            axes[1, 0].set_title(f'Mask {ids + 1} with Fitted Sinusoidal')
            axes[1, 0].plot(x_values, y_values, c='r', label='Contour of the Mask')
            axes[1, 0].plot(x_values, sinusoidal(x_values, *params), color='orange', label='Fitted Sinusoidal')

            # plot the ruler bars on top of the sinusoidal curve

            step_size = 1  # Adjust step size to plot every third point
            for i in range(0, len(upper_tangent_information), step_size):
                intersection, sinusoidal_point = upper_tangent_information[i]
                axes[1, 1].plot([sinusoidal_point[0], intersection[0]], [sinusoidal_point[1], intersection[1]],
                                color='green')
                axes[1, 1].scatter(intersection[0], intersection[1], color='green')
                axes[1, 1].scatter(sinusoidal_point[0], sinusoidal_point[1], color='blue')
            for i in range(0, len(lower_tangent_information), step_size):
                intersection, sinusoidal_point = lower_tangent_information[i]
                axes[1, 1].plot([sinusoidal_point[0], intersection[0]], [sinusoidal_point[1], intersection[1]],
                                color='orange')
                axes[1, 1].scatter(intersection[0], intersection[1], color='orange')
                axes[1, 1].scatter(sinusoidal_point[0], sinusoidal_point[1], color='blue')
                # invert the y-axis to have the same orientation as the borehole

            axes[1, 1].set_ylim(0, 360)
            axes[1, 1].invert_yaxis()

            axes[1, 1].set_ylim(0, 360)
            axes[1, 1].set_title(f'Sinusoidal curve and intersection points, ID: {ids}')
            axes[1, 1].legend()
            # Write down the calculated D and JRC values for upper and lower profiles between the upper and lower right plots

            text = (
                f" Mean aperture: {mean_aperture_mm :.2f} mm\nStd aperture: {std_aperture_mm:.2f} mm \nHigh-Side Azimuth and dip : {high_side_azimuth:.2f}°/{high_side_dip:.2f}°")

            # Add the text to the plot
            axes[1, 1].text(0.03, 0.05, f"{text}",
                            horizontalalignment='left', verticalalignment='bottom', transform=axes[1, 1].transAxes)
            plt.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, f'plot_Fourier_transform_structure{i}.png')
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path)
            plt.close()  # Close the figure to free up memory

            # plot of the curvature, overestimation fector and beta angle
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))

            num_images = len(image_segmentation_pairs)

            axes[0].plot(x_sin, y_sin, color='orange', label='fitting at the location of the fracture')
            axes[0].scatter(x_values, y_values, color='red', label='Fracture surface points', s=3)
            axes[0].set_title(f'sinsuoidal fit (ID {ids + 1})')
            axes[0].axhline(y=D, color='black', linestyle='--', label='Sinusoidal average point')
            # add labels to axes
            axes[0].set_xlabel('X [pixel]')
            axes[0].set_ylabel('Y [pixel]')

            axes[0].legend()

            axes[1].plot(x_tangents, curvatures, color='green')
            axes[1].set_title(f'curvature along the sinusoidal curve')
            # add labels to axes
            axes[1].set_xlabel('X [pixel]')
            axes[1].set_ylabel('Curvature')

            axes[1].legend()

            # if the correction factor shape is not equal to beta angle,  then leave one element form the beta_angle_vector out
            if len(beta_angle_vector) != len(correction_factors):
                beta_angle_vector = beta_angle_vector[:-1]
            axes[2].plot(beta_angle_vector, correction_factors, color='blue', label='correction factor')

            axes[2].plot(beta_angle_vector, over_estimation_factors, color='red', label='overestimation factor')
            axes[2].set_title(f'correction factor as function of beta angle')
            # add labels to axes
            axes[2].set_xlabel('Beta angle [°]')
            axes[2].set_ylabel('Factors')
            # set the x-axis with ticks every 50° but set the minimum and maximum automatically

            axes[2].legend()
            plt.legend()
            plt.tight_layout()
            file_path = os.path.join(output_dir, f'plot_beta_angle_correction_factor_curvature_dx_{dx}_structure{ids}_aperture{defined_aperture}.png')
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path)
            plt.close()  # Close the figure to free up memory

            # plot of the curvature, overestimation fector and beta angle
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            num_images = len(image_segmentation_pairs)

            axes[0].plot(x_sin_along_all_axis, y_sin_along_all_axis, color='grey', linestyle='--',
                         label='sinusoidal fitting along borehole perimeter')
            axes[0].plot(x_sin, y_sin, color='orange', label='fitting at the location of the fracture')
            axes[0].scatter(x_values, y_values, color='red', label='Fracture surface points', s=3)
            axes[0].set_title(f'sinsuoidal fit (ID {ids + 1})')
            axes[0].axhline(y=D, color='black', linestyle='--', label='Sinusoidal average point')
            # add labels to axes
            axes[0].set_xlabel('X [pixel]')
            axes[0].set_ylabel('Y [pixel]')

            axes[0].legend()

            axes[1].plot(x_sin, y_sin, color='orange', label='fitting at the location of the fracture')
            axes[1].scatter(x_values, y_values, color='red', label='Fracture surface points', s=3)
            axes[1].set_title(f'sinsuoidal fit (ID {ids + 1})')
            axes[1].axhline(y=D, color='black', linestyle='--', label='Sinusoidal average point')
            # add labels to axes
            axes[1].set_xlabel('X [pixel]')
            axes[1].set_ylabel('Y [pixel]')

            axes[1].legend()
    plt.legend()
    plt.tight_layout()
    file_path = os.path.join(output_dir, f'plot_fitting_local_and_along_all_axis_defined_dx_{dx}_aperture_{defined_aperture}.png')
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path)
    plt.close()  # Close the figure to free up memory







