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

# Set seed for reproducibility
SEED = 600
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

date_string = datetime.date.today().strftime("%m_%d")
data_directory = os.path.join(DATA_DIR, 'Reinforcement_Learning', 'Data_created_06_05_2024')

config = {
    "ch_lr_every": 10, "data_dir": data_directory, "validate_every": 15, "lr": [0.1], "ch_lr": 0.5,
    "momentum": 0.9, "use_nesterov": True, "smooth": 1.0, "batch_size": 20, "batch_size_val": 16,
    "model_dir": os.path.join(OUTPUT_DIR, 'Bedretto_models'), "test_set_IDs": []
}

create_directory(config["model_dir"])

config["model_name"] = '2D_unet_model' + date_string

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device used : ' + str(device))

model_name = '2D_unet_model'
forward_model = UNetOTV().to(device).float()

max_epochs = 350

# Collect all file IDs from the output data path
file_IDs = []
for filename in os.listdir(data_directory):
    if filename.endswith(".pt"):  # Assuming the files have a .pt extension
        file_id = int(filename.split("_")[1])  # Extract the ID number from the filename
        # Check if the ID is within the specified ranges for boreholes ST1 and MB8
        if (1 <= file_id <= 1650) or (1880 <= file_id <= 2309):
            file_IDs.append(os.path.join(data_directory, filename))

file_IDs_list = list(file_IDs)
# Randomly select 100 file IDs to be removed
removed_file_IDs_for_testing = set(random.sample(file_IDs_list, 100))

# Now you can proceed to create your dataset excluding these IDs
remaining_file_IDs = [file_id for file_id in file_IDs if file_id not in removed_file_IDs_for_testing]
complete_dataset = Dataset(remaining_file_IDs)

# Create your optimizer and scheduler
optimizer = optim.SGD(forward_model.parameters(), lr=config["lr"][0], momentum=config["momentum"], nesterov=config["use_nesterov"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

# Partition the dataset
train_size = int(np.ceil(len(complete_dataset) * 0.80))
val_size = int(np.floor(len(complete_dataset) * 0.2))
lengths = [train_size, val_size]

training_set, validation_set = torch.utils.data.random_split(complete_dataset, lengths, generator=torch.Generator().manual_seed(SEED))

loss_function = nn.BCELoss()
validation_loss = np.array([np.inf])

training_loader = data.DataLoader(training_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
validation_loader = data.DataLoader(validation_set, batch_size=config["batch_size_val"], shuffle=True)

step_no = 0
max_acc = 0

training_losses = list()
validation_losses = list()
best_validation_loss = float('inf')

# Loop over the epochs
for epoch in range(max_epochs):
    print('\nEpoch : ' + str(epoch))

    # Loop over the mini-batches
    for batch_idx, (image_in, index_mask) in enumerate(training_loader):
        forward_model.train()

        # Convert mask values from 1 to 0 and from 2 to 1
        index_mask = torch.where(index_mask == 1, 0, index_mask)  # Replace 1 with 0
        index_mask = torch.where(index_mask == 2, 1, index_mask)  # Replace 2 with 1

        image_in = image_in.to(device).float()
        index_mask = index_mask.to(device).float()

        optimizer.zero_grad()
        loss = loss_function(forward_model(image_in), index_mask)
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

        with torch.no_grad():
            for batch_idx, (image_in, index_mask) in enumerate(validation_loader):
                # Convert mask values from 1 to 0 and from 2 to 1
                index_mask = torch.where(index_mask == 1, 0, index_mask)  # Replace 1 with 0
                index_mask = torch.where(index_mask == 2, 1, index_mask)  # Replace 2 with 1
                image_in = image_in.to(device).float()
                index_mask = index_mask.to(device).float()

                prediction = forward_model(image_in)
                loss = loss_function(prediction, index_mask)
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
                       os.path.join(config["model_dir"], f"{config['model_name']}_with_all_handmade_labels_{epoch}_epochs.pt"))

# Save training and validation losses
config['training_losses'] = training_losses
config["validation_losses"] = validation_losses

import pickle

with open(os.path.join(config["model_dir"], f"{config['model_name']}_with_all_handmade_labels_{epoch}_epochs_config.p"), 'wb') as fp:
    pickle.dump(config, fp, protocol=pickle.HIGHEST_PROTOCOL)
