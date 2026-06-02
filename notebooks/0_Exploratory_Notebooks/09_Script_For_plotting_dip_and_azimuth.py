
import pandas as pd
import os
from deeplogger import DATA_DIR
import matplotlib.pyplot as plt
import numpy as np
#This script is only for plotting dips and azimuths for structures picked by experts, and segmented by the model

output_dir = 'BFE_05_final_model_geometrical_extraction'  # Directory to save the test results
os.makedirs(output_dir, exist_ok=True)



path_to_metadata_files = DATA_DIR + 'BFE_A_05_input_HS' + os.sep
metadata_files = pd.read_csv(path_to_metadata_files + 'BFE_A_05_metadata.csv', sep=',')


atv_visible_structures_files = pd.read_csv(path_to_metadata_files + 'BFE_A_05_composite_for_structural_comparison.csv', skiprows=[1])

# Access the columns
atv_visible_structures_depth = atv_visible_structures_files['Feature Depth']
atv_visible_structures_dip = atv_visible_structures_files['Dip']
atv_visible_structures_azimuth = atv_visible_structures_files['Azimuth']

# Filter only the structures labeled as fractures
picked_fractures = atv_visible_structures_files[atv_visible_structures_files['Type'] == 1]

picked_fractures_depth = picked_fractures['Feature Depth']
picked_fractures_dip = picked_fractures['Dip']
picked_fractures_azimuth = picked_fractures['Azimuth']


segmented_structures_1 = pd.read_csv(path_to_metadata_files + 'dip_azimuth_borehole_1_threshold_06.csv')

segmented_structures_1_depth = segmented_structures_1['Depth']
segmented_structures_1_dip = segmented_structures_1['Dip']
segmented_structures_1_azimuth = segmented_structures_1['Azimuth']

segmented_structures_2 = pd.read_csv(path_to_metadata_files + 'dip_azimuth_borehole_2_threshold_06.csv')

segmented_structures_2_depth = segmented_structures_2['Depth']
segmented_structures_2_dip = segmented_structures_2['Dip']
segmented_structures_2_azimuth = segmented_structures_2['Azimuth']


#start plotting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

ax1.scatter(atv_visible_structures_dip, atv_visible_structures_depth,  color='g', marker = 's', label='ATV visible structures (WellCAD)')
ax1.scatter( picked_fractures_dip, picked_fractures_depth,color='r', marker = '^', label='labeled fractures (WellCAD)')
ax1.scatter( segmented_structures_1_dip, segmented_structures_1_depth,color='black',marker = 'x', label='segmented')
ax1.scatter( segmented_structures_2_dip, segmented_structures_2_depth,color='black',marker = 'x', label='segmented')


ax1.set_ylabel('Depth [m]')
ax1.set_xlabel('Dip [°]')
ax1.tick_params(axis='y')

# Right y-axis for Azimuth
ax2.scatter( atv_visible_structures_azimuth, atv_visible_structures_depth, color='g',marker = 's', label='ATV visible structures (WellCAD)')
ax2.scatter( picked_fractures_azimuth, picked_fractures_depth,color='r', marker = '^', label='labeled fractures (WellCAD)')
ax2.scatter( segmented_structures_1_azimuth, segmented_structures_1_depth,color='black',marker = 'x', label='segmented')
ax2.scatter( segmented_structures_2_azimuth, segmented_structures_2_depth,color='black',marker = 'x', label='segmented')


ax2.set_xlabel('Azimuth [°]')
ax2.set_ylabel('')
#take out labels fro the ticks
ax2.set_yticklabels([])
ax2.tick_params(axis='y')

#set the x_lim to be in the range of all sin depths
ax1.set_ylim([np.min(picked_fractures_depth-2), np.max(picked_fractures_depth+2)])
ax1.set_xlim(0,90)
ax2.set_ylim([np.min(picked_fractures_depth-2), np.max(picked_fractures_depth+2)])
ax2.set_xlim(0,360)

ax1.invert_yaxis()
ax2.invert_yaxis()
#show legend on the ax 2, loc upper right

plt.legend(['picks visible on ATV*', 'picks labeled as fractures*', 'segmented structures'],
           loc='upper right', bbox_to_anchor=(0.95, 1.165))  # Adjust the coordinates as needed#add the legend for both plots outside the plot

# Add a title to the whole figure
fig.suptitle('Dip and Azimuth as a Function of Depth, Borehole BFE 05'
             '\n Classification Threshold: 0.6')
plt.show()
file_path = os.path.join(output_dir, 'Plot_picked_dip_az_vs_segmentations_4_at_scale_06.png')
if os.path.exists(file_path):
    os.remove(file_path)
fig.savefig(file_path, dpi=300)
plt.close()









