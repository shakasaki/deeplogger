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
from deeplogger.common_helpers import create_directory
import datetime


date_string = datetime.date.today().strftime("%m_%d")
data_directory = DATA_DIR + 'Bedretto_Output_NF_inverted' + os.sep

config = {"ch_lr_every": 10, "data_dir": data_directory, "validate_every": 20, "lr": [0.1], "ch_lr": 0.5,
          "momentum": 0.9, "use_nesterov": True, "smooth": 1.0, "batch_size": 20, "batch_size_val": 8,
          "model_dir": OUTPUT_DIR + 'Bedretto_models' + os.sep}

create_directory(config["model_dir"])

config["model_name"] = '2D_unet_model' + date_string


if(torch.cuda.is_available()):
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

for path in os.listdir(config["data_dir"] ):
    full_path = os.path.join(config["data_dir"] , path)
    if os.path.isfile(full_path):
        file_IDs.append(full_path)
        
        

complete_dataset = Dataset(file_IDs)

if(not os.path.isdir(config["model_dir"])):
    os.makedirs(config["model_dir"])

# create your optimizer
#    optimizer = optim.Adam(forward_model.parameters())
optimizer = optim.SGD(forward_model.parameters(), lr=config["lr"][0], momentum = config["momentum"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.75)

# Set training parameters


# partition the dataset
lengths = [int(np.ceil(len(complete_dataset)*0.8)), int(np.floor(len(complete_dataset)*0.2))]
training_set, validation_set = torch.utils.data.random_split(complete_dataset, lengths)
loss_function = DiceLoss()
validation_loss = np.array([np.inf])

training_loader = data.DataLoader(training_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
validation_loader = data.DataLoader(validation_set, batch_size=config["batch_size_val"], shuffle=True)


step_no = 0
max_acc = 0


def dice_coefficient(y_pred, y_true, smooth):
    y_pred = y_pred[:, 0].contiguous().view(-1)
    y_true = y_true[:, 0].contiguous().view(-1)
    intersection = (y_pred * y_true).sum()
    dsc = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dsc.item()


training_losses = list()
validation_losses = list()

for epoch in range(max_epochs):
    
    print('\nEpoch : ' + str(epoch))
    
    # Change learning rate
#    if epoch % config["ch_lr_every"] == 0 and epoch > 0:
#        config["lr"].append(config["lr"][-1] * config["ch_lr"])
#        optimizer = torch.optim.SGD(forward_model.parameters(), lr=config["lr"][-1],
#                nesterov=config["use_nesterov"], momentum=config["momentum"])
#        print("Optimizer update: learning rate changed to : {}".format(config["lr"]))   

    # loop over the mini-batches
    for batch_idx, (image_in, index_mask) in enumerate(training_loader):
        forward_model.train()
        image_in = image_in.to(device).float()
        index_mask = index_mask.to(device).long()
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
        accs = []
        for batch_idx, (image_in, index_mask) in enumerate(validation_loader):
            image_in = image_in.to(device).float()
            index_mask = index_mask.to(device).long()
            optimizer.zero_grad()
            prediction = forward_model(image_in)
            loss = loss_function(prediction, index_mask)
            optimizer.step()
            scheduler.step()
            validation_losses.append(loss.item())
            print("Validation loss : {}".format(loss.item()))
            accs.append(dice_coefficient(prediction, index_mask, config["smooth"]))
            avg_acc = np.mean(accs)
            print("final dev accuracy: {}".format(avg_acc))
            if avg_acc > max_acc:
                # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
                print("Saving best model in iteration {}".format(step_no))
                print("best validation accuracy: {}".format(avg_acc))
                max_acc = avg_acc
                torch.save(forward_model.state_dict(), os.path.join(config["model_dir"] + config["model_name"] + '-epoch-' + str(epoch) +'.pt'))
        
        
config['training_losses'] = training_losses
config["validation_losses"] = validation_losses

import pickle

with open(config["model_dir"] + config['model_name'] + '_config.p', 'wb') as fp:
    pickle.dump(config, fp, protocol=pickle.HIGHEST_PROTOCOL)

