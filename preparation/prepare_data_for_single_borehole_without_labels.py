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
from deeplogger.filters import *

# %%
image_rows = 360
selected_data_type = 'atv'
save_figures = True
Borehole_name = 'BFE_A_05'
Las_file_name = 'BFE_A_05_composite54.las'
date = 'unknown'

output_data_path = DATA_DIR + f'{Borehole_name}_data_snippets' + os.sep
path_to_folders = DATA_DIR + f'{Borehole_name}_input_HS' + os.sep
output_figures_path = OUTPUT_DIR + f'{Borehole_name}_Figures_snippets' + os.sep

create_directory(output_figures_path)
create_directory(output_data_path)

print("Output Data Path:", output_data_path)
print("Path to Folders:", path_to_folders)
print("Output Figures Path:", output_figures_path)

azimuth_values = 360


#Here, we define the metadata file which will contain the information about the snippets
metadata = pd.DataFrame(columns=['id', 'Borehole', 'Start Depth (m)', 'End Depth (m)', 'Data type', 'Date'])
# Create a new row to allow concatenation
new_row = pd.DataFrame({'id': [0], 'Borehole': ['MBX'], 'Start Depth (m)': [0], 'End Depth (m)': [0],
                        'Data type': [selected_data_type], 'Date': ['date']})
metadata = pd.concat([metadata, new_row], ignore_index=True)

## First, we create a loop to load the depths, depth resolutions, labels file, metadata of the borehole we want to export
#We proceed by doing it with the length of the file name, as we need to do that for every file of our selected data type

current_depth_correction = 0
depth_skip = 0
data_type = selected_data_type
# Use a different variable name, such as `current_borehole`, to avoid overwriting
path_to_data = path_to_folders + os.sep

# Them, we define the depth vector and the first data line of the file, the depth resolution and the number of snippets
depth_vector, first_dataline = get_depth_only(file_name=Las_file_name, data_path=path_to_data)
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
file_name = Las_file_name
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
        'Borehole': [Borehole_name],
        'Start Depth (m)': [depth_range[0]],
        'End Depth (m)': [depth_range[1]],
        'Data type': [data_type],
        'Date': [date]
    })

    # Concatenate the new row to the metadata DataFrame
    metadata = pd.concat([metadata, new_row], ignore_index=True)


    if save_figures:
        # Plotting the data_subset and image to visualize the created snippets
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        im=ax.imshow(data_subset, extent=[0, 360, np.max(depth_range), np.min(depth_range)], aspect='auto')
        ax.set_xlabel('Azimuth')
        ax.set_ylabel('Depth (m)')
        ax.set_title(f'Snippet {id}')

        fig.colorbar(label='Value', mappable=im, ax=ax)
        plt.savefig(output_figures_path + f"ID_SB_{id}_data.png")
        plt.close()
    pt.save([data_subset], output_data_path + f"ID_SB_HS_{id}_data.pt")

# remove first for from metadata dataframe
metadata = metadata.iloc[1:]
metadata.to_csv(path_to_folders + 'SB_boreholes_metadata.csv', index=False)
