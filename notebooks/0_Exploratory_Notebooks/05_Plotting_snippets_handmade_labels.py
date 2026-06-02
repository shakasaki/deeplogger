# This notebooks is just dedicated to plotting snippets of hand-made labels
import os
from deeplogger.dataloader import Dataset_np as Dataset
from deeplogger import DATA_DIR
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


atv_data_directory = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'atv_Data_created_13_05_2024' + os.sep
#load the snippet that I want to plot
#load snippet 37

file_IDs = []
for filename in os.listdir(atv_data_directory):
    if filename.endswith(".pt"):  # Assuming the files have a .pt extension
        file_id = int(filename.split("_")[1])  # Extract the ID number from the filename
        # Check if the ID is within the specified ranges for boreholes ST1 and MB8
        if (file_id == 37):
            file_IDs.append(os.path.join(atv_data_directory, filename))


Id = Dataset(file_IDs)
id_snippet = 37
output_dir = 'plots_presentation'
os.makedirs(output_dir, exist_ok=True)

cmap = ListedColormap(['none', 'orange'])
for data_subset, label in Id:
    # Move tensors to CPU
    data_subset_cpu = data_subset.cpu()
    label_cpu = label.cpu()

    plt.imshow(data_subset_cpu,cmap = 'viridis', aspect = 'auto')
    plt.imshow(label_cpu, alpha=0.5, cmap=cmap, aspect='auto')

    file_path = os.path.join(output_dir, f'plot_atv__id_{id_snippet}_cividis_cmap_2.png')

    # Check if the file already exists and delete it if it does
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)

    # Close the plot to free up memory
    plt.close()


# do the same for OTV data :

otv_data_directory = DATA_DIR + 'Reinforcement_Learning' + os.sep + 'Data_created_06_05_2024' + os.sep
#load the snippet that I want to plot
#load snippet 37

file_IDs = []
for filename in os.listdir(otv_data_directory):
    if filename.endswith(".pt"):  # Assuming the files have a .pt extension
        file_id = int(filename.split("_")[1])  # Extract the ID number from the filename
        # Check if the ID is within the specified ranges for boreholes ST1 and MB8
        if (file_id == 24):
            file_IDs.append(os.path.join(otv_data_directory, filename))


Id = Dataset(file_IDs)
id_snippet = 24
output_dir = 'plots_presentation'
os.makedirs(output_dir, exist_ok=True)

cmap = ListedColormap(['none', 'orange'])
for data_subset, label in Id:
    # Move tensors to CPU
    data_subset_cpu = data_subset.cpu()
    label_cpu = label.cpu()

    plt.imshow(data_subset_cpu,cmap = 'viridis', aspect = 'auto')
    plt.imshow(label_cpu, alpha=0.5, cmap=cmap, aspect='auto')

    file_path = os.path.join(output_dir, f'plot_atv_id_{id_snippet}_cividis_cmap_2.png')

    # Check if the file already exists and delete it if it does
    if os.path.exists(file_path):
        os.remove(file_path)

    # Save the plot
    plt.savefig(file_path)

    # Close the plot to free up memory
    plt.close()
























