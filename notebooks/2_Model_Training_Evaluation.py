import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
#import for setting the style of the plots
import matplotlib.style as style

# This notebook is to plot the training and validation losses of the models, as function of the eppochs


#path_to_file = OUTPUT_DIR + 'Bedretto_models'
# path_to_file_1 = r"C:\Users\StartKlar\Documents\Msc_Thesis\Models\ATV_models\WIth_Adam_optimizer"
path_to_file_1 = r"C:\Users\StartKlar\Documents\Msc_Thesis\Models\Evaluation"
# file_name_1 = '2D_unet_model05_28BCEloss_almost_all_ATV_batch_size_20_config.p'
file_name_1 = '2D_unet_model05_06BCEloss_all_ATV_batch_size_20_config.p'
# file_name_1 = '2D_unet_model04_18all_structures_training_BCEDice_300_epochs_config.p'
# file_name_1 = '2D_unet_model07_16BCEloss_all_ATV_adam_batch_size_20_config.p'

path_to_file_2 = r"C:\Users\StartKlar\Documents\Msc_Thesis\Models\Evaluation"
file_name_2 = '2D_unet_model04_25BCE_loss_all_dataset_inverted_labels_config.p'
file_path_1 = path_to_file_1 + os.sep + file_name_1
file_path_2 = path_to_file_2 + os.sep + file_name_2

# with open(file_path_1, 'rb') as file:
#     data = pickle.load(file)
# test_ids = data["test_set_IDs"]
# print(test_ids)

# Open the file and load the data
with open(file_path_1, 'rb') as file:
    data_1 = pickle.load(file)

with open(file_path_2, 'rb') as file:
    data_2 = pickle.load(file)
print(len(data_1['training_losses']))
print(len(data_1['validation_losses']))
validation_loss_1 = data_1['validation_losses']
training_loss_1 = data_1['training_losses']
epochs_training_1 = np.linspace(0, 600, len(training_loss_1))

epochs_validation_1 = range(20,600,20)


validation_loss_2 = data_2['validation_losses']
training_loss_2 = data_2['training_losses']
epochs_training_2 = np.linspace(0,500, len(training_loss_2))

epochs_validation_2 = range(15,484,15)



# fig, axs = plt.subplots(1, 2, figsize=(9, 4))
#
# # Plot the training loss in normal scale
# axs[0].plot(range(len(training)), training_loss_1, label='Training loss', color ='grey')
# axs[0].plot(epochs_validation_1, validation_loss_1, label='Validation loss', color = 'orange')
#
# axs[0].set_xlabel('Training epochs')
# axs[0].set_ylabel('log-Loss')
# # set axis to log scale
# axs[0].set_yscale('log')
# # Plot the validation loss in normal scale
# axs[1].plot(epochs_training_2, training_loss_2, label='Training loss', color ='grey')
# axs[1].plot(epochs_validation_2, validation_loss_2, label='Validation loss', color = 'orange')
# axs[1].set_xlabel('Training epochs')
# axs[1].legend(loc='upper right')
# axs[1].set_yscale('log')
# plt.tight_layout()
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()



fig, axs = plt.subplots(figsize=(8, 6))

# Plot the training loss
axs.plot(epochs_training_1, training_loss_1, label='Training loss', color='grey')
axs.plot(epochs_validation_1, validation_loss_1, label='Validation loss', color='orange')
axs.set_xlabel('Epochs')
axs.set_ylabel('log-Loss')
# set axis to log scale
axs.set_yscale('log')
# Remove y-axis numbers on the right plot
# axs[1].yaxis.set_tick_params(labelleft=False)

# Adjust the layout to make room for the title and legend
plt.subplots_adjust(top=0.85)

# Add a title to the plot
plt.suptitle('ATV training: 2D U-Net with BCE loss on automatically created labels, SGD optimizer', y=0.95)

# Define the legend for the plot
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2)
plt.show(dpi=300)





