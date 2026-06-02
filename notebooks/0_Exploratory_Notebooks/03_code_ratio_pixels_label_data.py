#This noetbook is used to calculate the ratio of labeled pixels per snippet image agaist the total number of pixels in the image. This is done to understand the distribution of labeled pixels in the dataset.

#import necessary packages
import os
import numpy as np
import torch
from deeplogger.dataloader import Dataset_np as Dataset
import matplotlib.pyplot as plt


#load the dataset (ATV)
data_path = r"C:\Users\StartKlar\Documents\Msc_Thesis\Data\Reinforcement_learning_labels\no_pre_trained_model_atv_05_13" + os.sep

file_IDs = []

for path in os.listdir(data_path):
    full_path = os.path.join(data_path, path)
    if os.path.isfile(full_path):
        file_IDs.append(full_path)

# Assuming complete_dataset is correctly initialized with only the first 10 files
complete_dataset = Dataset(file_IDs)  # Select only the first 10 files

# Initialize as empty lists
no_label_pixels = []
label_pixels = []
ratios_per_image = []

# Loop through the dataset
for image, indices in complete_dataset:
    # Move tensors to CPU if necessary
    image = image.cpu()
    indices = indices.cpu()
    indices = torch.where(indices == 1, 0, indices)  # Replace 1 with 0
    indices = torch.where(indices == 2, 1, indices)  # Replace 2 with 1
    indices = torch.where(indices == 3, 1, indices)  # Replace 3 with 1

    # For OTV, reshape image to a 2D tensor (height * width, channels)
    # image_flat = image.reshape(-1, image.shape[2])

    # Extract pixels where the label is 0
    # no_label_pixel_values = image_flat[indices.flatten() == 0].numpy()
    no_label_pixel_values = image[indices == 0].numpy()
    no_label_pixels.append(no_label_pixel_values)

    # Extract pixels where the label is 1
    # label_pixel_values = image_flat[indices.flatten() == 1].numpy()
    label_pixel_values = image[indices == 1].numpy()
    label_pixels.append(label_pixel_values)

    # Calculate the ratio for the current image
    if len(label_pixel_values) > 0:  # To avoid division by zero
        ratio = len(label_pixel_values) / len(no_label_pixel_values)
    else:
        ratio = 0 # Assign NaN if there are no label pixels to avoid division by zero
    ratios_per_image.append(ratio)

# Concatenate the lists of pixels into numpy arrays
no_label_pixels = np.concatenate(no_label_pixels) * 255
label_pixels = np.concatenate(label_pixels) * 255

# Convert the list of ratios to a numpy array
ratios_per_image = np.array(ratios_per_image)

mean_ratio = np.mean(ratios_per_image)
std_ratio = np.std(ratios_per_image)
median_ratio = np.median(ratios_per_image)


#Do the same but for OTV
data_path_otv = r"C:\Users\StartKlar\Documents\Msc_Thesis\Data\Reinforcement_learning_labels\no_pre_trained_model_05_06" + os.sep
file_IDs_otv = []

for path in os.listdir(data_path_otv):
    full_path = os.path.join(data_path_otv, path)
    if os.path.isfile(full_path):
        file_IDs_otv.append(full_path)

# Assuming complete_dataset_otv is correctly initialized with only the first 10 files
complete_dataset_otv = Dataset(file_IDs_otv)  # Select only the first 10 files

# Initialize as empty lists
no_label_pixels_otv = []
label_pixels_otv = []
ratios_per_image_otv = []

# Loop through the dataset
for image, indices in complete_dataset_otv:
    # Move tensors to CPU if necessary
    image = image.cpu()
    indices = indices.cpu()
    indices = torch.where(indices == 1, 0, indices)  # Replace 1 with 0
    indices = torch.where(indices == 2, 1, indices)  # Replace 2 with 1
    indices = torch.where(indices == 3, 1, indices)  # Replace 3 with 1

    image_flat = image.reshape(-1, image.shape[2])

    # Extract pixels where the label is 0
    no_label_pixel_values_otv = image_flat[indices.flatten() == 0].numpy()
    no_label_pixels_otv.append(no_label_pixel_values_otv)

    # Extract pixels where the label is 1
    label_pixel_values_otv = image_flat[indices.flatten() == 1].numpy()
    label_pixels_otv.append(label_pixel_values_otv)

    # Calculate the ratio for the current image
    if len(label_pixel_values_otv) > 0 and len(no_label_pixel_values_otv) > 0  :  # Check if there are any no label pixels
          # Check if there are any label pixels
        ratio_otv = len(label_pixel_values_otv) / len(no_label_pixel_values_otv)
    else:
            ratio_otv = 0  # Assign 0 if there are no label pixels in the image
    ratios_per_image_otv.append(ratio_otv)

# Concatenate the lists of pixels into numpy arrays
no_label_pixels_otv = np.concatenate(no_label_pixels_otv) * 255
label_pixels_otv = np.concatenate(label_pixels_otv) * 255

# Convert the list of ratios to a numpy array
ratios_per_image_otv = np.array(ratios_per_image_otv)
print(np.min(ratios_per_image_otv))
print(np.max(ratios_per_image_otv))
mean_ratio_otv = np.mean(ratios_per_image_otv)
std_ratio_otv = np.std(ratios_per_image_otv)
median_ratio_otv = np.median(ratios_per_image_otv)

# Plot the histograms
plt.hist(ratios_per_image, bins=1000, color='blue', alpha=0.3, label='ATV')
plt.hist(ratios_per_image_otv, bins=6000, color='orange', alpha=0.5, label='OTV')

# Add text with a bounding box for ATV
plt.text(
    2, 850,
    f'Mean ATV: {mean_ratio:.2f}\nMedian ATV: {median_ratio:.2f}\nStd ATV: {std_ratio:.2f}',
    bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'),
    fontsize=12
)
plt.text(
    1, 2200,
    f'Mean OTV: {mean_ratio_otv:.2f}\nMedian OTV: {median_ratio_otv:.2f}\nStd OTV: {std_ratio_otv:.2f}',
    bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.5'),
    fontsize=12
)
# Add text with a bounding box for OTV

# Set x-axis to log scale
plt.xscale('log')
plt.xlabel('Ratio of Labeled/Unlabeled Pixels')
plt.ylabel('Frequency')

plt.title('Distribution of Labeled Pixels Ratios')

plt.legend()

plt.show()




# Set x-axis to log scale
plt.xscale('log')
plt.xlabel('Ratio of Labeled/Unlabeled Pixels')
plt.ylabel('Frequency')

plt.title('Distribution of Labeled Pixels Ratios')

plt.legend()

plt.show()














