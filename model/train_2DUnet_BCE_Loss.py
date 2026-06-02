# import all the necessary packages
import os
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
from deeplogger.model_architectures import UNetOTV
from deeplogger.dataloader import Dataset_np as Dataset
from deeplogger.loss_functions import DiceLoss
import torch.utils.data as data
from deeplogger import DATA_DIR, OUTPUT_DIR
from deeplogger.common_helpers import create_directory
import datetime
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip


SEED = 600
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

date_string = datetime.date.today().strftime("%m_%d")
data_directory = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'Data_created_06_05_2024' + os.sep

config = {"ch_lr_every": 10, "data_dir": data_directory, "validate_every": 15, "lr": [0.1], "ch_lr": 0.5,
          "momentum": 0.9, "use_nesterov": True, "smooth": 1.0, "batch_size": 20, "batch_size_val": 16,
          "model_dir": OUTPUT_DIR + 'Bedretto_models' + os.sep, "test_set_IDs":[]}

create_directory(config["model_dir"])

config["model_name"] = '2D_unet_model' + date_string

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")
    torch.cuda.init()
else:
    device = torch.device("cpu")

print('Device used : ' + str(device))
model_name = '2D_unet_model'
forward_model = UNetOTV().to(device)
forward_model = forward_model.float()

max_epochs = 350

#The commented code below is not needed for the current task but it is to select a subset of the dataset
file_IDs = []


# Collect all file IDs from the output data path
for filename in os.listdir(data_directory):
    if filename.endswith(".pt"):  # Assuming the files have a .pt extension
        file_id = int(filename.split("_")[1])  # Extract the ID number from the filename
        # Check if the ID is within the specified ranges for boreholes ST1 and MB8
        if (1 <= file_id <= 1650) or (1880 <= file_id <= 2309):
            file_IDs.append(os.path.join(data_directory, filename))

# file_IDs = list()
#
# for path in os.listdir(config["data_dir"] ):
#     full_path = os.path.join(config["data_dir"] , path)
#     if os.path.isfile(full_path):
#         file_IDs.append(full_path)

file_IDs_list = list(file_IDs)

# Randomly select 100 file IDs to be removed
removed_file_IDs_for_testing = set(random.sample(file_IDs_list, 100))

# Now you can proceed to create your dataset excluding these IDs
remaining_file_IDs = [file_id for file_id in file_IDs if file_id not in removed_file_IDs_for_testing]
complete_dataset = Dataset(remaining_file_IDs)


# create your optimizer
#optimizer = optim.Adam(forward_model.parameters())
optimizer = optim.SGD(forward_model.parameters(), lr=config["lr"][0], momentum=config["momentum"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

# Set training parameters


# partition the dataset
train_size = int(np.ceil(len(complete_dataset) * 0.80))
val_size = int(np.floor(len(complete_dataset) * 0.2))

lengths = [train_size, val_size]

training_set, validation_set = torch.utils.data.random_split(complete_dataset, lengths, generator=torch.Generator().manual_seed(600))

loss_function = nn.BCELoss()
validation_loss = np.array([np.inf])

training_loader = data.DataLoader(training_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
validation_loader = data.DataLoader(validation_set, batch_size=config["batch_size_val"], shuffle=True)

step_no = 0
max_acc = 0

training_losses = list()
validation_losses = list()
best_validation_loss = float('inf')

# transform = Compose([
#     RandomHorizontalFlip(p=0.5)
# ])  # Set the probability of flipping to 0.5

# loop over the epochs
for epoch in range(max_epochs):
    print('\nEpoch : ' + str(epoch))

    # loop over the mini-batches
    for batch_idx, (image_in, index_mask) in enumerate(training_loader):
        forward_model.train()

        # Apply random horizontal flip to both image and mask
        # if random.random() > 0.5:  # Apply flip with 50% probability
        #     # Apply the same random horizontal flip to both image and mask
        #     image_in = transform(image_in)
        #     index_mask = transform(index_mask)

        # Convert mask values from 1 to 0 and from 2 to 1
        index_mask = torch.where(index_mask == 1, 0, index_mask)  # Replace 1 with 0
        index_mask = torch.where(index_mask == 2, 1, index_mask)  # Replace 2 with 1

        image_in = image_in.to(device).float()
        index_mask = index_mask.to(device).float()
        index_mask = index_mask.squeeze(0)

        optimizer.zero_grad()
        loss = nn.BCELoss()(forward_model(image_in), index_mask)
        loss.backward()
        training_losses.append(loss.item())
        optimizer.step()
        step_no += 1
        print("Training loss : {}".format(loss.item()))

    if epoch % config["validate_every"] == 0 and epoch > 0:
        torch.cuda.empty_cache()
        forward_model.eval()
        total_loss = 0
        total_batches = 0

        for batch_idx, (image_in, index_mask) in enumerate(validation_loader):
            # Convert mask values from 1 to 0 and from 2 to 1
            index_mask = torch.where(index_mask == 1, 0, index_mask)  # Replace 1 with 0
            index_mask = torch.where(index_mask == 2, 1, index_mask)  # Replace 2 with 1
            image_in = image_in.to(device).float()
            index_mask = index_mask.to(device).float()
            index_mask = index_mask.squeeze(0)

            optimizer.zero_grad()
            prediction = forward_model(image_in)
            loss = nn.BCELoss()(prediction, index_mask)
            total_loss += loss.item()
            total_batches += 1

        validation_loss = total_loss / total_batches
        validation_losses.append(validation_loss)
        print("Validation loss : {}".format(validation_loss))

        if validation_loss < best_validation_loss:
            # Save the best model
            best_validation_loss = validation_loss
            print("Saving best model in iteration {}".format(step_no))
            torch.save(forward_model.state_dict(),
                       os.path.join(
                           config["model_dir"] + config["model_name"] + 'with_all_handmade_labels' + str(epoch)+ '_epochs' + '.pt'))

# Save training and validation losses
config['training_losses'] = training_losses
config["validation_losses"] = validation_losses

import pickle

with open(config["model_dir"] + config['model_name'] +'with_all_handmade_labels' + str(epoch)+ '_epochs'+'_config.p', 'wb') as fp:
    pickle.dump(config, fp, protocol=pickle.HIGHEST_PROTOCOL)

