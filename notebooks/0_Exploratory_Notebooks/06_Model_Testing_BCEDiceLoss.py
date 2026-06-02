#Import the necesarry libraries
# import all the necessary packages
import os
import numpy as np
import torch
import torch.optim as optim
from deeplogger.model_architectures import UNetOTV
from deeplogger.dataloader import Dataset_np as Dataset
from deeplogger.loss_functions import DiceLoss
import torch.utils.data as data
from deeplogger import DATA_DIR, OUTPUT_DIR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import BCELoss

# Load the model saved model

# Load the model saved model
model_path = '/home/pperritaz/git/deeplogger/output/Bedretto_models/2D_unet_model04_18all_structures_training_BCEDice_500_epochs-epoch-280.pt'

model_name = '2D_unet_model04_18all_structures_training_BCEDice_500_epochs-epoch-280.pt'
# Create an instance of UNetOTV
model = UNetOTV()

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
#We define our BCE-Dice Loss function
class BCEDice_Loss(nn.Module):
    def __init__(self, forward_model):
        super(BCEDice_Loss, self).__init__()
        self.forward_model = forward_model
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, image_in, index_mask):
        y_pred = self.forward_model(image_in)
        return self.dice_loss(y_pred, index_mask) + self.bce_loss(y_pred, index_mask)

# Instantiate the loss function with the necessary argument
loss_function = BCEDice_Loss(model)

# start evaluating the model
#load a test dataset (here as we don't have the data, we will use the same training data but flipped horizontally)

data_directory = DATA_DIR + 'Bedretto_Output' + os.sep

file_IDs = list()

for path in os.listdir(data_directory):
    full_path = os.path.join(data_directory, path)
    if os.path.isfile(full_path):
        file_IDs.append(full_path)

complete_dataset = Dataset(file_IDs)

# only select files with at least one non-zero value


selected_batch_size = 8
# Create a DataLoader using the flipped dataset
test_loader = torch.utils.data.DataLoader(complete_dataset, batch_size=selected_batch_size, shuffle=True)

import matplotlib.pyplot as plt

def evaluate_model(model, testloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    num_correct = 0
    total_samples = 0
    image_segmentation_pairs = []  # List to store pairs of images and segmentation masks

    with torch.no_grad():
        for images, indices in testloader:
            images = images.to(device).float()  # Convert input to float
            indices = indices.to(device).float()

            # Forward pass
            outputs = model(images)  # Forward pass through the model
            loss = criterion(images, indices)  # Compute loss using custom criterion
            test_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(outputs)  # Assuming binary segmentation
            num_correct += (predicted == indices).sum().item()
            total_samples += indices.numel()

            # Convert predicted segmentation to match the format of indices (0 and 1)
            segmentation = predicted.byte()  # Convert to byte tensor (0 and 1)

            # Combine images and segmentation masks into pairs and append to the list
            image_segmentation_pairs.extend(zip(images.cpu(), segmentation.cpu(), indices.cpu()))

    # Compute the average test loss
    test_loss /= len(testloader)

    # Compute the test accuracy
    test_accuracy = num_correct / total_samples

    return test_loss, test_accuracy, image_segmentation_pairs



# Call the function to evaluate the model
test_loss, test_accuracy, image_segmentation_pairs = evaluate_model(model=model, testloader=test_loader, criterion=loss_function, device=device)

# Call the function to evaluate the model
# test_loss, test_accuracy, num_well_predicted, total_samples = evaluate_model(model=model, testloader=test_loader, criterion=DiceLoss(), device=device)
# Create a confusion matrix :
print( 'for the same dataset as the training dataset :' )
print('Test Loss : ', test_loss)
print('Test Accuracy : ', test_accuracy)

# Here we want to the first image and the first segmentation tensor
#first, we create a diectory to save the plots and then visualize them


# Define the output directory
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

num_pairs_to_plot = 20  # Change this value to plot more or fewer pairs
for i, (image, segmentation_mask, original_mask) in enumerate(image_segmentation_pairs[:num_pairs_to_plot]):
    plt.figure(figsize=(15, 5))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(0,1,2))  # Assuming image is in CHW format
    plt.title('Original Image')
    plt.axis('off')

    # Plot the segmentation mask overlaid on the original image
    plt.subplot(1, 3, 2)
    plt.imshow(image.permute(0, 1, 2))
    plt.imshow(segmentation_mask.squeeze().cpu().numpy(), cmap='copper')
    plt.title('Segmentation Mask Overlay')
    plt.axis('off')

    # Plot the original mask (label)
    plt.subplot(1, 3, 3)
    plt.imshow(original_mask.squeeze().cpu().numpy(), cmap='copper')
    plt.title('Original Mask (Label)')
    plt.axis('off')

    # Construct the file path
    file_path = os.path.join(output_dir, f'plot_{model_name}_pair{i}.png')

    # Check if the file already exists and delete it if it does
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)

    # Close the plot to free up memory
    plt.close()



#Now we want to test our model for a vertically flipped dataset
def flip_image_and_mask(image, mask):
    # Always flip the image and mask
    image = torch.flip(image, dims=[0])
    mask = torch.flip(mask, dims=[0])
    return image, mask

# Create a flipped dataset
flipped_dataset = []

for image, mask in complete_dataset:
    flipped_image, flipped_mask = flip_image_and_mask(image, mask)
    flipped_dataset.append((flipped_image, flipped_mask))

test_loader_flipped = torch.utils.data.DataLoader(flipped_dataset, batch_size=selected_batch_size, shuffle=True)
#
# # # Call the function to evaluate the model
test_loss, test_accuracy, image_segmentation_pairs = evaluate_model(model=model, testloader=test_loader_flipped, criterion=loss_function, device=device)


print( 'for the same dataset as the training dataset :' )
print('Test Loss : ', test_loss)
print('Test Accuracy : ', test_accuracy)

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

num_pairs_to_plot = 20  # Change this value to plot more or fewer pairs
for i, (image, segmentation_mask, original_mask) in enumerate(image_segmentation_pairs[:num_pairs_to_plot]):
    plt.figure(figsize=(15, 5))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(0,1,2))  # Assuming image is in CHW format
    plt.title('Original Image')
    plt.axis('off')

    # Plot the segmentation mask overlaid on the original image
    plt.subplot(1, 3, 2)
    plt.imshow(image.permute(0, 1, 2))
    plt.imshow(segmentation_mask.squeeze().cpu().numpy(), cmap='copper')
    plt.title('Segmentation Mask Overlay')
    plt.axis('off')

    # Plot the original mask (label)
    plt.subplot(1, 3, 3)
    plt.imshow(original_mask.squeeze().cpu().numpy(), cmap='copper')
    plt.title('Original Mask (Label)')
    plt.axis('off')
    # Construct the file path
    file_path = os.path.join(output_dir, f'plot_{model_name}_pair{i}_flipped.png')

    # Check if the file already exists and delete it if it does
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)

    # Close the plot to free up memory
    plt.close()






















