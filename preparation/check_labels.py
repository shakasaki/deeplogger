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

# %%
image_rows = 360
save_figures = False
# Define the path to the data folders
path_to_folders = DATA_DIR + 'Bedretto_Input_HS' + os.sep
output_figures_path = OUTPUT_DIR + 'Bedretto_Figures' + os.sep
output_data_path = DATA_DIR + 'Bedretto_Output' + os.sep

create_directory(output_figures_path)
create_directory(output_data_path)

azimuth_values = 360
azimuth_correction = 0
dip_correction = 0

# Load the information file which contains the file names, data types, borehole names, borehole diameters, etc.
information_files = pd.read_excel(path_to_folders + 'file_informations.xlsx')

#only load the data type we want to work with ('otv'), if we want to prepare  for the atv data, just change it by 'atv'
information_files = information_files[information_files['data type'] == 'otv']
#then, we only select the lines which have the data type we want to work with

#again, if we want to change it to the atv data, just change it by 'atv. from here, nothing else needs to be changed
selected_data_type = 'otv'
borehole = information_files["borehole"].to_numpy()
file_names = information_files["file name"].to_numpy()
date = information_files["date"].to_numpy()
depth_correction = information_files["depth correction"].to_numpy()
label_file_name = information_files["label file name"].to_numpy()
data_type = information_files["data type"].to_numpy()
bh_diameter = information_files["bh  diameter [m]"].to_numpy()


index = 1
file_name = file_names[index]
# Use a different variable name, such as `current_borehole`, to avoid overwriting
current_borehole = borehole[index]
path_to_data = path_to_folders + os.sep + current_borehole + os.sep
current_depth_correction = depth_correction[index]

# Then, we define the depth vector and the first data line of the file, the depth resolution and the number of snippets
depth_vector, first_dataline = get_depth_only(file_name=file_name, data_path=path_to_data)
corrected_depth_vector = depth_vector - current_depth_correction
depth_resolution = np.mean(np.abs(np.diff(corrected_depth_vector)))
image_size = (image_rows, 360)
# reshape the depth vector into n by image_rows matrix, skipping any incomplete rows
corrected_depth_vector = corrected_depth_vector[
                         :len(corrected_depth_vector) - len(corrected_depth_vector) % image_rows]
corrected_depth_matrix = corrected_depth_vector.reshape(int(corrected_depth_vector.shape[0] / image_rows),
                                                        image_rows)

# Here, we wall the file containing the labels (there is one per borehole, in the same folder as the data file), its name is defined in the excel sheet containing the information)
label_file = path_to_data + label_file_name[index] + '.xlsx'
label_df = pd.read_excel(label_file)



# if azimuth is 0 in the labels assign it to a value of 4
label_df['Azimuth'][label_df['Azimuth'] == 0] = 4

# create a big matrix with all borehole labels to start with
borehole_labels = np.zeros((corrected_depth_vector.shape[0], azimuth_values), dtype=int)
for i, label in label_df.iterrows():
    frac = Fracture(azimuth=label['Azimuth'], dip=label['Dip'],
                    depth=label['Depth'], aperture=label['Aperture'])
    borehole_labels = apply_label(image_in=borehole_labels,
                                  fracture=frac,
                                  depth_vector=corrected_depth_vector,
                                  bh_diameter=bh_diameter[index],
                                  azimuth_values=azimuth_values)

labels_conc = np.sum(borehole_labels, axis=1)
plt.plot(corrected_depth_vector, labels_conc)
plt.xlabel('Depth [m]')
plt.ylabel('Number of labels')
plt.title('Number of labels per depth')
plt.show()

# check for nans
nans_labels = np.sum(np.isnan(borehole_labels))

# Apply gaussian blur and check for nans
blurred_labels = gaussian_blur(borehole_labels[10000:10360,:], kernel_size=15)
nans_blurred_labels = np.sum(np.isnan(blurred_labels))
plt.imshow(blurred_labels, aspect='auto')
plt.show()

plt.imshow(borehole_labels[10000:10360,:])
plt.show()
