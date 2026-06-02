# In this script, is imported the pre-trained ResNet101 model from the torchvision library.
# Tests are performed on the model to check if it is working correctly.


## Here are information about the model : (from https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.fcn_resnet101.html#torchvision.models.segmentation.FCN_ResNet101_Weights)
#The inference transforms are available at FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms and perform the following preprocessing operations: Accepts PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects.
# The images are resized to resize_size=[520] using interpolation=InterpolationMode.BILINEAR. Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].


# import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import datetime
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image
from deeplogger import DATA_DIR
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


date_string = datetime.date.today().strftime("%m_%d")
model_name = 'Pre_trained_ResNET101' + date_string

# Load the pre-trained ResNet-101 model
model = fcn_resnet101(weights=FCN_ResNet101_Weights.DEFAULT, progress=True)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

# Define the custom segmentation head
class CustomSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomSegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Replace the classification head with the custom segmentation head
num_classes = 1  # Background and Foreground
dummy_input = torch.randn(1, 3, 520, 520)  # Assuming input size 520x520 and 3 channels (RGB)
dummy_input = dummy_input.to(device)
backbone_output = model.backbone(dummy_input)
backbone_output_shape = backbone_output['out'].shape
in_channels = backbone_output_shape[1]  # Assuming the channels are the second dimension
model.classifier = CustomSegmentationHead(in_channels, num_classes)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy loss for segmentation
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Convert model parameters to Double
model = model.double()

# Convert model weights to Double
for param in model.parameters():
    param.data = param.data.cuda().double()

# Load the data and create the DataLoader
output_data_path = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'Data_created_06_05_2024' + os.sep
file_IDs = [os.path.join(output_data_path, filename) for filename in os.listdir(output_data_path) if filename.endswith(".pt")]
complete_dataset = CustomDataset(file_IDs, transform=load_and_preprocess_data)
complete_dataset = [(image, torch.where(indices == 1, 0, torch.where(indices == 2, 1, torch.where(indices == 3, 1, indices)))) for image, indices in complete_dataset]

#split the dataset into training and validation datasets
lengths = [int(np.ceil(len(complete_dataset) * 0.8)), int(np.floor(len(complete_dataset) * 0.2))]
training_set, validation_set = torch.utils.data.random_split(complete_dataset, lengths)

training_loader = DataLoader(training_set, batch_size=8, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=4, shuffle=True)

# Define variables to store training losses
from sklearn.metrics import jaccard_score

# Define variables to store training and validation losses and metrics
training_losses = []
validation_losses = []
training_iou_scores = []
validation_iou_scores = []
training_dice_scores = []
validation_dice_scores = []

output_dir = 'training_pre_trained_models'
os.makedirs(output_dir, exist_ok=True)

# Training loop
num_epochs = 35
for epoch in range(num_epochs):
    model.train()
    running_training_loss = 0.0

    # Initialize variables to calculate IoU and Dice coefficient for training set
    training_intersection = 0
    training_union = 0
    training_total_predicted_positive = 0
    training_total_true_positive = 0

    # Training phase
    for images, labels in training_loader:
        optimizer.zero_grad()

        # Convert the input data and labels to DoubleTensor and move to GPU
        images = images.to(torch.double).to(device)
        labels = labels.to(torch.double).to(device)

        outputs = model(images)
        loss = criterion(outputs['out'], labels)
        loss.backward()
        optimizer.step()
        running_training_loss += loss.item()

        predicted = torch.sigmoid(outputs['out']) > 0.5
        predicted = predicted.to(torch.int)  # Convert predicted to IntTensor for bitwise operation
        labels = labels.to(torch.int)  # Convert labels to IntTensor for bitwise operation

        # Calculate IoU and Dice coefficient for training set
        training_intersection += torch.sum((predicted & labels)).item()  # Perform bitwise AND without converting to int
        training_union += torch.sum((predicted | labels)).item()  # Perform bitwise OR without converting to int
        training_total_predicted_positive += torch.sum(predicted).item()
        training_total_true_positive += torch.sum(labels).item()

    # Calculate training epoch loss
    training_epoch_loss = running_training_loss / len(training_loader.dataset)
    training_losses.append(training_epoch_loss)

    # Calculate training IoU and Dice coefficient
    training_iou = training_intersection / training_union
    training_dice = 2 * training_intersection / (training_total_predicted_positive + training_total_true_positive)
    training_iou_scores.append(training_iou)
    training_dice_scores.append(training_dice)

    # Validation phase
    model.eval()
    running_validation_loss = 0.0

    # Initialize variables to calculate IoU and Dice coefficient for validation set
    validation_intersection = 0
    validation_union = 0
    validation_total_predicted_positive = 0
    validation_total_true_positive = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            # Convert the input data and labels to DoubleTensor and move to GPU
            images = images.to(torch.double).to(device)
            labels = labels.to(torch.double).to(device)

            outputs = model(images)
            loss = criterion(outputs['out'], labels)
            running_validation_loss += loss.item()

            # Calculate IoU and Dice coefficient for validation set
            predicted = torch.sigmoid(outputs['out']) > 0.5
            predicted = predicted.to(torch.int)  # Convert predicted to IntTensor for bitwise operation
            labels = labels.to(torch.int)  # Convert labels to IntTensor for bitwise operation

            # Calculate IoU and Dice coefficient for validation set
            validation_intersection += torch.sum((predicted & labels)).item()  # Perform bitwise AND without converting to int
            validation_union += torch.sum((predicted | labels)).item()  # Perform bitwise OR without converting to int
            validation_total_predicted_positive += torch.sum(predicted).item()
            validation_total_true_positive += torch.sum(labels).item()
    # Calculate validation epoch loss
    validation_epoch_loss = running_validation_loss / len(validation_loader.dataset)
    validation_losses.append(validation_epoch_loss)

    # Calculate validation IoU and Dice coefficient
    validation_iou = validation_intersection / validation_union
    validation_dice = 2 * validation_intersection / (validation_total_predicted_positive + validation_total_true_positive)
    validation_iou_scores.append(validation_iou)
    validation_dice_scores.append(validation_dice)

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Training Loss: {training_epoch_loss:.4f}, "
          f"Validation Loss: {validation_epoch_loss:.4f}, "
          f"Training IoU: {training_iou:.4f}, "
          f"Validation IoU: {validation_iou:.4f}, "
          f"Training Dice Coefficient: {training_dice:.4f}, "
          f"Validation Dice Coefficient: {validation_dice:.4f}")

    # Check if the current epoch's validation loss is less than the best loss encountered so far
    if epoch % 5 == 4:  # Check every 5 epochs
        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss
            # Save the model
            model_save_path = os.path.join(output_dir, f'Pre_trained_resnet101_{date_string}_{epoch + 1}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {validation_epoch_loss:.4f}")
# Save the metrics to a file

with open(os.path.join(output_dir, f'metrics_fine_tuned_resnet101_{date_string}_{num_epochs}.txt'), 'w') as f:
    f.write("Epoch\tTrain Loss\tVal Loss\tTrain IoU\tVal IoU\tTrain Dice\tVal Dice\n")
    for i in range(num_epochs):
        f.write(f"{i+1}\t{training_losses[i]}\t{validation_losses[i]}\t{training_iou_scores[i]}\t{validation_iou_scores[i]}\t{training_dice_scores[i]}\t{validation_dice_scores[i]}\n")

#
# # Plot the training losses and accuracies
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(training_losses, color = 'blue', label="Training Loss")
# plt.plot(validation_losses, color = 'red', label="Validation Loss")
# plt.title("Training and Validation Losses")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
#
# plt.subplot(1, 2, 2)
# plt.plot(training_iou_scores, color = 'blue', label="Training IoU")
# plt.plot(training_dice_scores, color = 'green', label="Training Dice")
# plt.plot(validation_iou_scores, color = 'red', label="Validation IoU")
# plt.plot(validation_dice_scores, color = 'red', label="Validation Dice")
# plt.title("Training and Validation Metrics")
# plt.xlabel("Epoch")
# plt.ylabel("IoU/Dice Coefficient")
# plt.legend()
#
# # Construct the file path
# file_path = os.path.join(output_dir, f'{model_name}_{date_string}_{num_epochs}epochs_train_loss_accuracy.png')
#
# # Save the plot
# plt.savefig(file_path)
#
# # Close the plot to free up memory
# plt.close()
#
# Save the fine-tuned model
torch.save(model.state_dict(), os.path.join(output_dir, f'Pre_trained_resnet101_{date_string}_{num_epochs}.pt'))
print("Model saved successfully.")


# #test the model
# # Load the test data
# test_data_path = DATA_DIR + os.sep + 'Bedretto_Test_NF'
# test_file_IDs = []
#
#
# # Collect all file IDs from the output data path
# for filename in os.listdir(output_data_path):
#     if filename.endswith(".pt"):  # Assuming the files have a .pt extension
#         file_id = int(filename.split("_")[1])  # Extract the ID number from the filename
#         # Check if the ID is within the specified ranges for boreholes ST1 and MB8
#         if (0 <= file_id <= 365) or (1117 <= file_id <= 2086) or (file_id >= 2310):
#             file_IDs.append(os.path.join(output_data_path, filename))
#
# dataset = CustomDataset(file_IDs, transform=load_and_preprocess_data)
# test_loader = DataLoader(dataset, batch_size=8, shuffle=True)
#
# #test the model
# model.eval()
# correct_pixels = 0
# total_pixels = 0
# correct_images = 0
# total_images = 0
#
# # Enforce CPU computation for model predictions
# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.double()
#         labels = labels.double()
#
#         # Ensure CPU computation for model outputs
#         outputs = model(images.device)
#
#         # Calculate per-pixel accuracy
#         predicted_labels = (outputs['out'] > 0.5).float()  # Threshold predictions
#         correct_pixels += (predicted_labels == labels).sum().item()
#         total_pixels += labels.numel()
#
#         # Calculate per-image accuracy
#         correct_images += ((predicted_labels == labels).sum(dim=(1, 2, 3)) == labels.numel()).sum().item()
#         total_images += labels.size(0)
#
# #plot the resulting segmentation masks
# plt.figure(figsize=(10, 5))
# for i in range(4):
#     plt.subplot(2, 4, i + 1)
#     plt.imshow(to_pil_image(images[i].byte()))
#     plt.title("Input Image")
#     plt.axis('off')
#
#     plt.subplot(2, 4, i + 5)
#     plt.imshow(to_pil_image(images[i].byte()))
#     plt.imshow(predicted_labels[i].squeeze().cpu().numpy(), cmap='gray', alpha = 0.5)
#     plt.title("Predicted Mask")
#     plt.axis('off')
#
# # Construct the file path
# file_path = os.path.join(output_dir, f'{model_name}_{date_string}_{num_epochs}_epochs_test_predictions.png')
#
# # Save the plot
# plt.savefig(file_path)
#
# # Close the plot to free up memory
# plt.close()
#
#



























