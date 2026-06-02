import pandas as pd
import torch as pt
from deeplogger import DATA_DIR, OUTPUT_DIR
from deeplogger.common_helpers import create_directory
from deeplogger.importLASv3 import *
from deeplogger.image_processing import *
from skimage.transform import resize
import os
from deeplogger.image_processing import *
from deeplogger.labels import *
from deeplogger.labels import apply_label
from deeplogger.config import Fracture
from deeplogger.filters import *

# %%
image_rows = 360
save_figures = True
gaussian_blur_flag = False
neighbor_filter_applied = False
invert_label_applied = False
minimum_aperture = 15
selected_data_type = 'atv'


# Define the path to the data folders
if selected_data_type == 'atv':
    output_data_path = DATA_DIR + 'Bedretto_Output_ATV' + os.sep
    path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
    output_figures_path = OUTPUT_DIR + 'Bedretto_Figures_ATV' + os.sep
elif neighbor_filter_applied and invert_label_applied:
    output_data_path = DATA_DIR + 'Bedretto_Output_NF_inverted' + os.sep
    path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
    output_figures_path = OUTPUT_DIR + 'Bedretto_Figures_NF_inverted' + os.sep
elif neighbor_filter_applied:
    output_data_path = DATA_DIR + 'Bedretto_Output_NF' + os.sep
    path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
    output_figures_path = OUTPUT_DIR + 'Bedretto_Figures_NF' + os.sep
elif invert_label_applied:
    output_data_path = DATA_DIR + 'Bedretto_Output_inverted' + os.sep
    path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
    output_figures_path = OUTPUT_DIR + 'Bedretto_Figures_inverted' + os.sep
elif gaussian_blur_flag:
    output_data_path = DATA_DIR + 'Bedretto_Output_GB' + os.sep
    path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
    output_figures_path = OUTPUT_DIR + 'Bedretto_Figures_GB' + os.sep

else:
    output_data_path = DATA_DIR + 'Bedretto_Output' + os.sep
    path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
    output_figures_path = OUTPUT_DIR + 'Bedretto_Figures' + os.sep

create_directory(output_figures_path)
create_directory(output_data_path)

print("Output Data Path:", output_data_path)
print("Path to Folders:", path_to_folders)
print("Output Figures Path:", output_figures_path)

azimuth_values = 360
azimuth_correction = 0
dip_correction = 0

# Load the information file which contains the file names, data types, borehole names, borehole diameters, etc.
information_files = pd.read_excel(path_to_folders + 'file_informations.xlsx')

#only load the data type we want to work with ('otv'), if we want to prepare  for the atv data, just change it by 'atv'
information_files = information_files[information_files['data type'] == 'atv']
selected_data_type = 'atv'
# keep only the boreholes with the selected data type
information_files = information_files[information_files['data type'] == selected_data_type]

columns_labels = ["Depth", "Azimuth", "Dip", "Aperture", "Bedretto_Structure"]
#Here, we define the metadata file which will contain the information about the snippets
metadata = pd.DataFrame(columns=['id', 'Borehole', 'Start Depth (m)', 'End Depth (m)', 'Data type', 'Date'])
# Create a new row to allow concatenation
new_row = pd.DataFrame({'id': [0], 'Borehole': ['MBX'], 'Start Depth (m)': [0], 'End Depth (m)': [0],
                        'Data type': [selected_data_type], 'Date': ['date']})
metadata = pd.concat([metadata, new_row], ignore_index=True)

## First, we create a loop to load the depths, depth resolutions, labels file, metadata of the borehole we want to export
#We proceed by doing it with the length of the file name, as we need to do that for every file of our selected data type
for index, row in information_files.iterrows():
    file_name = row['file name']
    current_borehole = row['borehole']
    date = row["date"]
    current_depth_correction = row["depth correction"]
    depth_skip = row["Skip depth [m]"]
    label_file_name = row["label file name"]
    data_type = row["data type"]
    bh_diameter = row["bh  diameter [m]"]
    # Use a different variable name, such as `current_borehole`, to avoid overwriting
    path_to_data = path_to_folders + current_borehole + os.sep

    # Them, we define the depth vector and the first data line of the file, the depth resolution and the number of snippets
    depth_vector, first_dataline = get_depth_only(file_name=file_name, data_path=path_to_data)
    shifted_depth_vector = depth_vector - current_depth_correction

    # Skip first part
    shifted_depth_vector = shifted_depth_vector[shifted_depth_vector > depth_skip]

    depth_resolution = np.mean(np.abs(np.diff(shifted_depth_vector)))
    image_size = (image_rows, 360)
    # reshape the depth vector into n by image_rows matrix, skipping any incomplete rows
    corrected_depth_vector = shifted_depth_vector[
                             :len(shifted_depth_vector) - len(shifted_depth_vector) % image_rows]
    corrected_depth_matrix = corrected_depth_vector.reshape(int(corrected_depth_vector.shape[0] / image_rows),
                                                            image_rows)

    # Here, we wall the file containing the labels (there is one per borehole, in the same folder as the data file), its name is defined in the excel sheet containing the information)
    label_file = path_to_data + label_file_name + '.xlsx'
    label_df = pd.read_excel(label_file)

    # if azimuth is 0 in the labels assign it to a value of 4
    if minimum_aperture is not None:
        label_df['Aperture'].iloc[label_df['Aperture'] <= minimum_aperture] = minimum_aperture

    # create a big matrix with all borehole labels to start with
    borehole_labels = np.zeros((corrected_depth_vector.shape[0], azimuth_values), dtype=int)
    for index, label in label_df.iterrows():
        frac = Fracture(azimuth=label['Azimuth'], dip=label['Dip'],
                        depth=label['Depth'], aperture=label['Aperture'])
        borehole_labels = apply_label(image_in=borehole_labels,
                                      fracture=frac,
                                      depth_vector=corrected_depth_vector,
                                      bh_diameter=bh_diameter,
                                      azimuth_values=azimuth_values)

    # create a large label image for the borehole
    for index, depth_subset in enumerate(corrected_depth_matrix):
        # define the parameters to get the data_subset for the current range
        depth_range = [depth_subset.min(), depth_subset.max()]
        data_subset, depth_index = get_data_subset_from_depth_range(
            file_name=file_name,
            data_path=path_to_data,
            depth=corrected_depth_vector,
            data_type=selected_data_type,
            depth_range=depth_range
        )
        # # For the 'otv' data type, the azimuthal is not the same for all the boreholes, so we need to resize the data to have the same number of azimuth values for all the boreholes, namely 360
        # if data_subset.shape[1] != 360:
        #     data_subset = resize(data_subset, (data_subset.shape[0], azimuth_values),
        #                          anti_aliasing=True)
        #for the 'atv' data, we also assume this, but for both width and height of the image (some problems were encouneterd during gthe ML)
        if data_subset.shape != (360, 360):
            data_subset = resize(data_subset, (360, 360), anti_aliasing=True)

        #Here, we just replace the -9999 values by 0 for each snippet
        data_subset = replace_empty_measurements(data_subset, 0)

        # for atv data, we apply a svd filtering to the data subset
        if selected_data_type == 'atv':
            data_subset, svd_decomp = remove_svd(data_subset, low_s=0, high_s=2)

        # here, we save the metadata in of the current snippet in the metadata file
        # Increment the ID
        id = metadata['id'].iloc[-1] + 1
        # Construct a new DataFrame row with scalar values
        new_row = pd.DataFrame({
            'id': [id],
            'Borehole': [current_borehole],
            'Start Depth (m)': [depth_range[0]],
            'End Depth (m)': [depth_range[1]],
            'Data type': [data_type],
            'Date': [date]
        })

        # Concatenate the new row to the metadata DataFrame
        metadata = pd.concat([metadata, new_row], ignore_index=True)
        label_index, index_values = get_index_from_depth_range(corrected_depth_vector, depth_range)
        label_subset = borehole_labels[label_index, :]
        # invert image along x axis (TODO: why??)
        label_subset = np.flip(label_subset, axis=1)
        # Ensure the image has the right dimensions

        #if you want to apply a kernel blur to the image, you can use the following line :
        if gaussian_blur_flag:
            label_subset = gaussian_blur(label_subset, kernel_size=15)
        #if you want to apply a neighbor filter to the image, you can use the following line :
        if neighbor_filter_applied:
            label_subset = neighbor_filter(label_subset, kernel_size=3)
        if invert_label_applied:
            label_subset = invert_label_values(label_subset)
        if save_figures:
            # Plotting the data_subset and image to visualize the created snippets
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            axs[0].imshow(data_subset, extent=[0, 360, np.max(depth_range), np.min(depth_range)], aspect='auto')
            im0 = axs[0].imshow(label_subset, extent=[0, 360, np.max(depth_range), np.min(depth_range)], alpha=0.2,
                                aspect='auto',
                                vmin=0,
                                vmax=1,)
            axs[0].set_xlabel('Azimuth')
            axs[0].set_ylabel('Depth (m)')
            axs[0].set_title(f'Snippet {id}')
            fig.colorbar(label='Value', ax=axs[0], mappable=im0)
            im1 = axs[1].imshow(label_subset, extent=[0, 360, np.max(depth_range), np.min(depth_range)], aspect='auto',
                                cmap='grey',
                                vmax=1,
                                vmin=0)
            axs[1].set_xlabel('Azimuth')
            axs[1].set_ylabel('Depth (m)')
            axs[1].set_title(f'Label {id}')
            fig.colorbar(label='Value', mappable=im1, ax=axs[1])
            plt.savefig(output_figures_path + f"ID_{id}_data_label.png")
            plt.close()
        pt.save([data_subset, label_subset], output_data_path + f"ID_{id}_data_label.pt")

# remove first for from metadata dataframe
metadata = metadata.iloc[1:]
metadata.to_csv(path_to_folders + 'Bedretto_metadata.csv', index=False)
