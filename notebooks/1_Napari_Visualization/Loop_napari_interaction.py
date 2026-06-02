#import the necessary packages

import napari

from deeplogger import DATA_DIR, OUTPUT_DIR

import napari

from glob import glob

import os

from qtpy.QtWidgets import QPushButton

import torch as pt

import datetime

from deeplogger.common_helpers import create_directory

import matplotlib.pyplot as plt

import numpy as np

##load the data and labels from the file

# data_path =DATA_DIR + 'Bedretto_Output'
data_path = r'C:\Users\StartKlar\Documents\Msc_Thesis\Data\SB_borehole_analysis_for_model_testing\label_creation'
image_files = glob(data_path + os.sep + '*data.pt') # looks for all files with this ending in the directory

#
# Define the directory where the new labels are stored
#
# # Function to load and process remaining data label files
#
# # Define the directory where the new labels are stored
# new_labels_dir = r'C:\Users\StartKlar\Documents\Msc_Thesis\Data\Reinforcement_learning_labels\no_pre_trained_model_...'
#
# # Get the list of new label files
# new_label_files = glob(new_labels_dir + os.sep + '*.pt')
#
# # Extract the numbers from the filenames of the new labels
# existing_numbers = [int(os.path.basename(file).split('_')[1]) for file in new_label_files]
#
# # Define the directory where all the data labels are stored
# all_labels_dir = r'C:\Users\StartKlar\Documents\Msc_Thesis\Data\ATV_data_labels'
#
# # Get the list of all data label files
# all_label_files = glob(all_labels_dir + os.sep + '*_label.pt')
#
# # Filter out the data label files that have already been processed
# image_files = [file for file in all_label_files if int(os.path.basename(file).split('_')[1]) not in existing_numbers]

# Define the viewer as napari
viewer = napari.Viewer()

# To keep track of what we do, we are going to save the new labels with a name containing the date of creation and the pre-trained model used
pre_trained_model = 'no_pre_trained_model'
date_string = datetime.date.today().strftime("%m_%d")

new_data_path = r'C:\Users\StartKlar\Documents\Msc_Thesis\Data\SB_borehole_analysis_for_model_testing\label_creation' + os.sep + f'SB23_15_SB31_08_handmade_labels_{date_string}'
create_directory(new_data_path)

## Here we define necessary functions

def update_image(viewer, image_path):
# Load the new image
        image = pt.load(image_path)


# Check if the loaded data is a list, and convert it to a numpy array if necessary
        if isinstance(image, list):
            image = np.array(image)
        else:
            image = image
# Update or add the image layer

        if 'Image' in viewer.layers:
            viewer.layers['Image'].data = image
        else:
            viewer.add_image(image, name='Image')

def save_labels(image, labels_layer, image_path):
# save the image and label together in a pytorch array, but in a folder for reinforcement learning
    file_name = os.path.basename(image_path).split('.')[0]  # Extract file name without extension
    save_path = os.path.join(new_data_path, f"{file_name}_atv_new.pt")
    pt.save((image.data, labels_layer.data), save_path)

def main(image_files):

    viewer = napari.Viewer()

    btn = QPushButton("Next Image")

    viewer.window.add_dock_widget(btn)

    image_iter = iter(image_files)

    current_image_path = next(image_iter)

    update_image(viewer, current_image_path)


    def load_next_image():

        nonlocal current_image_path #changed 'nonlocal' by global, otehrwise there is an error
        save_labels(viewer.layers['Image'], viewer.layers['Labels'], current_image_path)

        try:
            current_image_path = next(image_iter)
            update_image(viewer, current_image_path)

        except StopIteration:
            viewer.close()



    btn.clicked.connect(load_next_image)

    napari.run()

## start processing the images
main(image_files)



# load the saved arrays and plot the labels

label_files = glob(new_data_path + '*label_SB.pt')


output_dir = r'C:\Users\StartKlar\Documents\Msc_Thesis\Data\SB_borehole_analysis_for_model_testing\Figures' + os.sep + f'Handmade_labels_SB23_15_SB31_08_{date_string}'
create_directory(output_dir)

for label_file in label_files:
    # Load the saved arrays
    image, labels = pt.load(label_file)

    # Plotting
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(labels)
    ax[1].set_title('Labels')

    # Save the plot
    save_path = os.path.join(output_dir, os.path.basename(label_file).split('.')[0] + '.png')
    plt.savefig(save_path)

    # Close the plot to free up memory
    plt.close()














