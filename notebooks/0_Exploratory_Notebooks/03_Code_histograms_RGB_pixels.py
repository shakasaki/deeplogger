#first, load the snippets and the labels
from deeplogger import DATA_DIR
import os
import pandas as pd
import matplotlib.pyplot as plt
from deeplogger.dataloader import Dataset_np as Dataset
import numpy as np
import torch
from torch.utils import data

data_path = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'Data_created_06_05_2024' + os.sep
# data_path = r'C:\Users\StartKlar\Documents\Msc_Thesis\Data\Reinforcement_learning_labels\no_pre_trained_model_05_06' + os.sep
file_IDs = list()

for path in os.listdir(data_path):
    full_path = os.path.join(data_path , path)
    if os.path.isfile(full_path):
        file_IDs.append(full_path)

complete_dataset = Dataset(file_IDs)
# only select the first 10 files
# Initialize as empty numpy arrays
no_label_pixels = []
label_pixels = []
# Loop through the dataset
# Loop through the dataset
for image, indices in complete_dataset:
    # Move tensors to CPU if necessary
    image = image.cpu()
    indices = indices.cpu()
    indices = torch.where(indices == 1, 0, indices)  # Replace 1 with 0
    indices = torch.where(indices == 2, 1, indices)  # Replace 2 with 1

    # Reshape image to a 2D tensor (height * width, channels)
    image_flat = image.reshape(-1, image.shape[2])

    # Extract pixels where the label is 0
    no_label_pixels.append(image_flat[indices.flatten() == 0])

    # Extract pixels where the label is 1
    label_pixels.append(image_flat[indices.flatten() == 1])

# Concatenate the lists of pixels into numpy arrays
no_label_pixels = np.concatenate(no_label_pixels) * 255
label_pixels = np.concatenate(label_pixels) * 255

#create a directory to save the files :
output_dir = 'histograms_RGB_data'
os.makedirs(output_dir, exist_ok=True)

# Calculate mean, median, and std for pixels where label is not present
no_label_mean = np.mean(no_label_pixels, axis=0)
no_label_median = np.median(no_label_pixels, axis=0)
no_label_std = np.std(no_label_pixels, axis=0)

# Calculate mean, median, and std for pixels where label is present
label_mean = np.mean(label_pixels, axis=0)
label_median = np.median(label_pixels, axis=0)
label_std = np.std(label_pixels, axis=0)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot histograms for pixels where label is not present
ax[0].hist(no_label_pixels[:, 0], bins=50, color='red', alpha=0.5)
ax[0].hist(no_label_pixels[:, 1], bins=50, color='green', alpha=0.5)
ax[0].hist(no_label_pixels[:, 2], bins=50, color='blue', alpha=0.5)
ax[0].set_title('RGB values of pixels in the background')
ax[0].set_xlabel('Pixel Value')
ax[0].set_ylabel('log Frequency')
ax[0].legend()
ax[0].set_yscale('log')  # Set y-axis to log scale
ax[0].text(0.95, 0.05, f"Mean: R: {no_label_mean[0]:.1f}, G: {no_label_mean[1]:.1f}, B: {no_label_mean[2]:.1f}\nMedian: R: {no_label_median[0]:.1f}, G: {no_label_median[1]:.1f}, B: {no_label_median[2]:.1f}\nStd: R: {no_label_std[0]:.1f}, G: {no_label_std[1]:.1f}, B: {no_label_std[2]:.1f}", transform=ax[0].transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

# Plot histograms for pixels where label is present
ax[1].hist(label_pixels[:, 0], bins=50, color='red', alpha=0.5, label='Red')
ax[1].hist(label_pixels[:, 1], bins=50, color='green', alpha=0.5, label='Green')
ax[1].hist(label_pixels[:, 2], bins=50, color='blue', alpha=0.5, label='Blue')
ax[1].set_title('RGB values of pixels where label is present')
ax[1].set_xlabel('Pixel Value')
ax[1].legend()
ax[1].set_yscale('log')  # Set y-axis to log scale
ax[1].text(0.95, 0.05, f"Mean: R: {label_mean[0]:.1f}, G: {label_mean[1]:.1f}, B: {label_mean[2]:.1f}\nMedian: R: {label_median[0]:.1f}, G: {label_median[1]:.1f}, B: {label_median[2]:.1f}\nStd: R: {label_std[0]:.1f}, G: {label_std[1]:.1f}, B: {label_std[2]:.1f}", transform=ax[1].transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()  # Adjust layout to prevent overlapping

plt.show()
file_path = os.path.join(output_dir, 'histograms_first_created_dataset_by_hand.png')

# Check if the file already exists and delete it if it does
if os.path.exists(file_path):
    os.remove(file_path)

# Save the plot
plt.savefig(file_path)

# Close the plot to free up memory
plt.close()
