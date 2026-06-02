# this script is to conduct model testing on my local machine
#first, we import the necessary packages

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
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet101
from torch.nn import BCELoss
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms.functional import to_pil_image
from matplotlib.colors import Normalize


class CustomSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomSegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Set the seed for reproducibility
model = fcn_resnet101(pretrained=False, progress=False)

# Replace the classification head with the custom segmentation head
num_classes = 1  # Background and Foreground
backbone_output = model.backbone(torch.randn(1, 3, 520, 520))  # Assuming input size 520x520 and 3 channels (RGB)
backbone_output_shape = backbone_output['out'].shape
in_channels = backbone_output_shape[1]  # Assuming the channels are the second dimension
model.classifier = CustomSegmentationHead(in_channels, num_classes)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model.to(device)
else:
    device = torch.device("cpu")


# Load the saved model state dictionary
model_dir = '/home/pperritaz/git/deeplogger/model/training_pre_trained_models/'
model_file = 'fine_tuned_resnet101_05_02_20.pt'  # Adjust the filename as needed
saved_model_path = os.path.join(model_dir, model_file)
# Load the saved model state dictionary
saved_model_state = torch.load(saved_model_path, map_location=torch.device('cuda:0'))
model_state = model.state_dict()

# Filter out unexpected keys
saved_model_state = {k: v for k, v in saved_model_state.items() if k in model_state}

# Update the model state dictionary
model_state.update(saved_model_state)

# Load the updated state dictionary into the model
model.load_state_dict(model_state)
# # Load the saved model state dictionary

# # Define loss function and optimizer
# criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy loss for segmentation
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Convert model parameters to Double
model = model.double()

# Convert model weights to Double
for param in model.parameters():
    param.data = param.data.to(device).double()

model = model.to(device)

# Put the model in evaluation mode
model.eval()

# Define a custom dataset class to load the data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data, label = torch.load(file_path)
        if self.transform:
            data, label = self.transform(data, label)
        return data, label

# Define a function to load and preprocess data
def load_and_preprocess_data(data, label):
    # Convert data from HWC to CHW format
    data = torch.from_numpy(data).permute(2, 0, 1)
    # Convert label to tensor
    label = torch.from_numpy(label).unsqueeze(1). permute(1, 0, 2)
    return data, label

# Load the test data
test_data_path = data_directory = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'Data_created_06_05_2024'+ os.sep


test_file_IDs = []
#
# # Collect all file IDs from the output data path
# for filename in os.listdir(test_data_path):
#     if filename.endswith(".pt"):  # Assuming the files have a .pt extension
#         file_id = int(filename.split("_")[1])  # Extract the ID number from the filename
#         # Check if the ID is within the specified ranges for boreholes ST1 and MB8
#         if (366 <= file_id <= 1118) or (2087 <= file_id <= 2309):
#             test_file_IDs.append(os.path.join(test_data_path, filename))

dataset = CustomDataset(test_file_IDs, transform=load_and_preprocess_data)
test_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Test the model
model.eval()
correct_pixels = 0
total_pixels = 0
correct_images = 0
total_images = 0

output_dir = 'plots_segmentation_masks'
os.makedirs(output_dir, exist_ok=True)
# Initialize variables to store evaluation metrics
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Iterate through the test data loader
for images, labels in test_loader:
    # Move data to the appropriate device
    images = images.to(device)
    labels = labels.to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(images)
        predicted_labels = (outputs['out'] > 0.5).float()

    # Update evaluation metrics
    true_positives += ((predicted_labels == 1) & (labels == 1)).sum().item()
    false_positives += ((predicted_labels == 1) & (labels == 0)).sum().item()
    true_negatives += ((predicted_labels == 0) & (labels == 0)).sum().item()
    false_negatives += ((predicted_labels == 0) & (labels == 1)).sum().item()

# Calculate evaluation metrics
accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)



# Define a function to visualize the input image and predicted mask
def plot_segmentation(image, predicted_mask):
    plt.figure(figsize=(10, 5))

    # Plot input image
    plt.subplot(1, 2, 1)
    plt.imshow(to_pil_image(image.cpu()))
    plt.title("Input Image")
    plt.axis('off')

    # Plot predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(to_pil_image(image.cpu()))
    plt.imshow(predicted_mask.squeeze().cpu().numpy(), cmap='gray', alpha=0.5)
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()

# Iterate through the test data loader
for images, labels in test_loader:
    # Move input tensors to the device
    images = images.to(device)
    labels = labels.to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(images)
        predicted_labels = (outputs['out'] > 0.1).float()

    # Visualize results for the first batch
    for i in range(len(images)):
        plot_segmentation(images[i], predicted_labels[i])
        # Save the plots in the output directory
        # Add colorbar
        plt.colorbar(norm=Normalize(vmin=0, vmax=1))
        plt.savefig(os.path.join(output_dir, f"segmentation_{i}.png"))
        plt.close()

