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
import torch.nn.functional as F


SEED = 51
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

date_string = datetime.date.today().strftime("%m_%d")
data_directory = DATA_DIR + 'Bedretto_Output_NF_inverted' + os.sep

config = {"ch_lr_every": 10, "data_dir": data_directory, "validate_every": 20, "lr": [0.005], "ch_lr": 0.5,
          "momentum": 0.9, "use_nesterov": True, "smooth": 1.0, "batch_size": 20, "batch_size_val": 8,
          "model_dir": OUTPUT_DIR + 'Bedretto_models' + os.sep}


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

max_epochs = 300
file_IDs = list()

for path in os.listdir(config["data_dir"]):
    full_path = os.path.join(config["data_dir"], path)
    if os.path.isfile(full_path):
        file_IDs.append(full_path)

# only select ids until 1915 (Here we are using a part of the dataset where there is not too much mud on the background)
complete_dataset = Dataset(file_IDs[:1915])

# create your optimizer
optimizer = optim.Adam(forward_model.parameters())
#optimizer = optim.SGD(forward_model.parameters(), lr=config["lr"][0], momentum=config["momentum"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

# Set training parameters


# partition the dataset
lengths = [int(np.ceil(len(complete_dataset) * 0.8)), int(np.floor(len(complete_dataset) * 0.2))]
training_set, validation_set = torch.utils.data.random_split(complete_dataset, lengths)

#here we create a conbination of BCE and Dice loss (from Alexakis & Armenakis, 2020)
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

# Instantiate the loss function with the necessary argument
loss_function = BCEDiceLoss()

validation_loss = float('inf') #this was also changed

training_loader = data.DataLoader(training_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
validation_loader = data.DataLoader(validation_set, batch_size=config["batch_size_val"], shuffle=True)

step_no = 0
max_acc = 0


# def dice_coefficient(y_pred, y_true, smooth):
#     y_pred = y_pred[:, 0].contiguous().view(-1)
#     y_true = y_true[:, 0].contiguous().view(-1)
#     intersection = (y_pred * y_true).sum()
#     dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
#     return dsc.item()


training_losses = list()
validation_losses = list()
best_validation_loss = float('inf')

transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5)
])  # Set the probability of flipping to 0.5
# loop over the epochs
for epoch in range(max_epochs):
    print('\nEpoch : ' + str(epoch))

    # Training phase
    forward_model.train()
    for batch_idx, (image_in, index_mask) in enumerate(training_loader):

        # Apply random horizontal flip to both image and mask
        if random.random() > 0.5:  # Apply flip with 50% probability
            # Apply the same random horizontal flip to both image and mask
            image_in = transform(image_in)
            index_mask = transform(index_mask)

        image_in = image_in.to(device).float()
        index_mask = index_mask.to(device).float()
        index_mask = index_mask.squeeze(0)

        optimizer.zero_grad()
        prediction = forward_model(image_in)
        loss = loss_function(prediction, index_mask)
        loss.backward()
        training_losses.append(loss.item())
        optimizer.step()
        step_no += 1
        print("Training loss : {}".format(loss.item()))

    # Validation phase
    if epoch % config["validate_every"] == 0 and epoch > 0:
        torch.cuda.empty_cache()
        forward_model.eval()
        total_loss = 0
        total_batches = 0

        for batch_idx, (image_in, index_mask) in enumerate(validation_loader):
            image_in = image_in.to(device).float()
            index_mask = index_mask.to(device).float()
            index_mask = index_mask.squeeze(0)

            optimizer.zero_grad()
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
                       os.path.join(
                           config["model_dir"] + config["model_name"] + 'all_structures_training_BCEDice_500_epochs-epoch-' + str(epoch) + '.pt'))

# Save training and validation losses
config['training_losses'] = training_losses
config["validation_losses"] = validation_losses

import pickle

with open(config["model_dir"] + config['model_name'] +'all_structures_training_BCEDice_500_epochs'+ '_config.p', 'wb') as fp:
    pickle.dump(config, fp, protocol=pickle.HIGHEST_PROTOCOL)

