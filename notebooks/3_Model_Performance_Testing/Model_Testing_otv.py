# This script was used to vizualise the segmentations created on saved models trained on OTV data (Unsuccessful models).
#Import the necesarry libraries

import os
import numpy as np
import torch
import torch.optim as optim
from deeplogger.model_architectures import UNetOTV
from deeplogger.dataloader import Dataset_np as Dataset
import torch.utils.data as data
from deeplogger import DATA_DIR, OUTPUT_DIR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import BCELoss
import seaborn as sns
from matplotlib.colors import ListedColormap
import pickle
# Load the model saved model

# Load the model saved model
model_name = '2D_unet_model05_16with_all_handmade_labels30_epochs.pt'
model_path = '/home/pperritaz/git/deeplogger/output/Bedretto_models' + os.sep + model_name


# Create an instance of UNetOTV

# Define the necessary arguments
in_channels = 3  # Number of input channels (RGB image)
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

# start evaluating the model
#load a test dataset (here as we don't have the data, we will use the same training data but flipped horizontally)

data_directory = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'Data_created_06_05_2024' + os.sep #replace it with your data directory

# Here, we import the testing IDs :
config_file = '/home/pperritaz/git/deeplogger/output/Bedretto_models' + os.sep + '2D_unet_model05_16with_all_handmade_labels599_epochs_config.p'

with open(config_file, 'rb') as file:
    data = pickle.load(file)

# Use the 'data' variable to access the 'test_set_IDs'
test_ids = data["test_set_IDs"]
# only load the files corresponding to the test_ids


file_IDs = list()

# Loop through the files in the data directory
for path in os.listdir(data_directory):
    # Extract the file ID from the path
    file_id = path.split('.')[0]  # Adjust the splitting logic based on your file naming convention

    # Check if the file ID is in the list of test IDs
    if file_id in test_ids:
        full_path = os.path.join(data_directory, path)
        if os.path.isfile(full_path):
            file_IDs.append(full_path)

# Now file_IDs contains only the paths of the files that are in the test set
complete_dataset = Dataset(file_IDs)

selected_batch_size = 8


# Replace 1 with 0 and 2 with 1 in the dataset (the labels are 1 and 2, we want to convert them to 0 and 1)
complete_dataset = [(image, torch.where(indices == 1, 0, torch.where(indices == 2, 1, torch.where(indices == 3, 1, indices)))) for image, indices in complete_dataset]

# Create a DataLoader using the flipped dataset
test_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=selected_batch_size, shuffle=True)
# Inside the evaluation loop, right after loading the batch
#create a loop where the indices are replaced by 0 and 1

def evaluate_model(model, testloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_losses = []
    num_correct = 0
    total_samples = 0
    image_segmentation_pairs = []  # List to store pairs of images and segmentation masks
    pixel_f1_scores = []
    pixel_accuracies = []
    confusion_matrix = np.zeros((2, 2))  # Initialize the confusion matrix

    with torch.no_grad():
        for images, indices in testloader:
            images = images.to(device).float()  # Convert input to float
            indices = indices.to(device).float()  # Add channel dimension and convert target to float

            # Forward pass
            outputs = model(images)

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
            indices = indices.byte()
            # Compute confusion matrix
            tp = torch.sum(segmentation & indices).item()
            fp = torch.sum(segmentation & ~indices).item()
            fn = torch.sum(~segmentation & indices).item()
            tn = torch.sum(~segmentation & ~indices).item()

            confusion_matrix[1, 1] += tp
            confusion_matrix[0, 1] += fp
            confusion_matrix[1, 0] += fn
            confusion_matrix[0, 0] += tn

            # Calculate F1 score
            smooth = 1e-5
            intersection = (segmentation & indices).sum().float()
            union = (segmentation | indices).sum().float()
            pixel_f1 = (2.0 * intersection + smooth) / (segmentation.sum() + indices.sum() + smooth)
            pixel_f1_scores.append(pixel_f1)

            # Calculate pixel-wise accuracy
            correct_pixels = (segmentation == indices).sum().float()
            total_pixels = segmentation.numel()
            pixel_accuracy = correct_pixels / total_pixels
            pixel_accuracies.append(pixel_accuracy)

            # Combine images and segmentation masks into pairs and append to the list
            image_segmentation_pairs.extend(zip(images.cpu(), segmentation.cpu(), indices.cpu()))

        # Compute the average test loss
        test_loss = sum(test_losses) / len(test_losses)

        # Compute the test accuracy
        test_accuracy = num_correct / total_samples

        # Compute the average pixel-wise F1 score and accuracy
        avg_pixel_f1 = torch.mean(torch.stack(pixel_f1_scores))
        avg_pixel_accuracy = torch.mean(torch.stack(pixel_accuracies))

    return test_loss, test_accuracy, avg_pixel_f1.item(), avg_pixel_accuracy.item(), confusion_matrix, image_segmentation_pairs


def plot_confusion_matrix(confusion_matrix):
    # Define class labels
    classes = ['Negative', 'Positive']

    # Create heatmap
    sns.set(font_scale=1.4)  # Adjust font size
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    return plt


#Now we want to test our model for a vertically flipped dataset
def flip_image_and_mask(image, indices):
    # Always flip the image and mask
    image = torch.flip(image, dims=[0])
    indices = torch.flip(indices, dims=[0])
    return image, indices


# Call the function to evaluate the model
test_loss, test_accuracy,avg_pixel_f1, avg_pixel_accuracy, confusion_matrix, image_segmentation_pairs = evaluate_model(model=model, testloader=test_loader, criterion=BCELoss(), device=device)

# Print the results
print( 'for an unseen dataset :' )
print('Test Loss : ', test_loss)
print('Test Accuracy : ', test_accuracy)
print('Average pixel-wise F1 score : ', avg_pixel_f1)
print('Average pixel-wise accuracy : ', avg_pixel_accuracy)


## plot 40 images of test dataset against the model and save them in an output directory called 'plots'

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

num_images = 70  # Change this value to plot more or fewer pairs
cmap = ListedColormap(['none', 'red'])
for i, (image, segmentation, label) in enumerate(image_segmentation_pairs[:num_images]):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

        # Display the image
    ax[0].imshow(image.squeeze().cpu().numpy(), cmap='viridis')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

        # Overlay the label using the custom colormap
        # 'none' will be used for 0 values, making them transparent
    ax[1].imshow(image.squeeze().cpu().numpy(), cmap='viridis')
    ax[1].imshow(label.squeeze().cpu().numpy(), cmap=cmap, alpha = 0.7)
    ax[1].set_title('Original Label')
    ax[1].axis('off')

            # Overlay the segmentation mask using the custom colormap
    ax[2].imshow(image.squeeze().cpu().numpy(), cmap='viridis')
    ax[2].imshow(segmentation.squeeze().cpu().numpy(), cmap=cmap, alpha = 0.7)
    ax[2].set_title('Generated Segmentation Mask')
    ax[2].axis('off')

#save the plots on the distant server
    #onstruct the file path
    file_path = os.path.join(output_dir, f'plot_{model_name}_flipped_same_dataset_pair{i}_.png')

    # Check if the file already exists and delete it if it does
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)

    # Close the plot to free up memory
    plt.close(fig)














