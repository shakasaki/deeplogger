#This scirpt is used to evaluate the model on the unseen dataset; for an entire borheole (if images are concatenated, line 117 & 118)
#It was used to calculate the evaluation metrics on SB 2.2 and 2.3 borehoels


# import all the necessary packages

import os
import torch
from deeplogger.model_architectures_ATV import UNetOTV
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from deeplogger import DATA_DIR
from torch.nn import BCELoss
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, adjusted_rand_score, mutual_info_score, precision_recall_curve, auc
import seaborn as sns
import pickle
import pandas as pd
import cv2
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


class Dataset_np(data.Dataset):

    def __init__(self, list_IDs):
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        image_id = self.list_IDs[index]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        sample = torch.load(image_id)
        image = torch.tensor(sample[0], device=device)
        indices = torch.tensor(sample[1], device=device)
        return image, indices, image_id


#define the evaluation loop :

def evaluate_model(model, testloader, criterion, device, threshold):
    model.eval()
    test_losses = []
    num_correct = 0
    total_samples = 0
    image_segmentation_pairs = []
    pixel_f1_scores = []
    pixel_accuracies = []
    iou_scores = []
    dice_scores = []
    precisions = []
    recalls = []
    rand_errors = []
    variations_of_information = []
    specificities = []
    accuracies = []

    y_true = []
    y_scores = []

    confusion_matrix = np.zeros((2, 2))

    with torch.no_grad():
        for images, indices, image_id in test_loader:
            images = images.to(device).float()
            indices = indices.unsqueeze(1).to(device).float()

            outputs = model(images)
            if len(outputs.shape) == 2:
                outputs = outputs.unsqueeze(1)

            indices = indices.view_as(outputs)
            loss = criterion(outputs, indices)
            test_losses.append(loss.item())

            # Save the predicted probabilities
            predicted_probs = outputs.cpu().numpy()

            predicted = (outputs >= threshold).float()

            #put all images on top of each other, so the metrics can be calculated for the whole dataset
            if total_samples == 0:
                all_predicted = predicted
                all_indices = indices
            else:
                all_predicted = torch.cat((all_predicted, predicted), dim=0)
                all_indices = torch.cat((all_indices, indices), dim=0)



            num_correct += (all_predicted == all_indices).sum().item()
            total_samples += all_indices.numel()

            segmentation = all_predicted.byte()
            indices = all_indices.byte()

            smooth = 1e-5
            intersection = (segmentation & indices).sum().float()
            union = (segmentation | indices).sum().float()
            iou = (intersection + smooth) / (union + smooth)
            dice = 2.0 * intersection / (segmentation.sum() + indices.sum() + smooth)
            iou_scores.append(iou.item())
            dice_scores.append(dice.item())

            tp = torch.sum(segmentation & indices).item()
            fp = torch.sum(segmentation & ~indices).item()
            fn = torch.sum(~segmentation & indices).item()
            tn = torch.sum(~segmentation & ~indices).item()

            confusion_matrix[1, 1] += tp
            confusion_matrix[0, 1] += fp
            confusion_matrix[1, 0] += fn
            confusion_matrix[0, 0] += tn

            confusion_matrix = np.around(confusion_matrix, decimals=3)

            #prevent division by 0
            if tp + fn == 0:
                sensitivity = 0
            else:
                sensitivity = tp / (tp + fn)

            if tn + fp == 0:
                specificity = 0
            else:
                specificity = tn / (tn + fp)
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            if tp + tn + fp + fn == 0:
                accuracy = 0
            else:
                accuracy = (tp + tn) / (tp + tn + fp + fn)


            intersection = (segmentation & indices).sum().float()
            union = (segmentation | indices).sum().float()
            pixel_f1 = (2.0 * intersection + smooth) / (segmentation.sum() + indices.sum() + smooth)
            pixel_f1_scores.append(pixel_f1)

            correct_pixels = (segmentation == indices).sum().float()
            total_pixels = segmentation.numel()
            pixel_accuracy = correct_pixels / total_pixels
            pixel_accuracies.append(pixel_accuracy)

            precisions.append(precision)
            recalls.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)

            precisions.append(precision_score(indices.cpu().numpy().flatten(), predicted.cpu().numpy().flatten()))
            recalls.append(recall_score(indices.cpu().numpy().flatten(), predicted.cpu().numpy().flatten()))
            rand_errors.append(adjusted_rand_score(indices.cpu().numpy().flatten(), predicted.cpu().numpy().flatten()))
            variations_of_information.append(mutual_info_score(indices.cpu().numpy().flatten(), predicted.cpu().numpy().flatten()))

            # Save the image, segmentation, indices, and predicted probabilities
            image_segmentation_pairs.extend(zip(images.cpu(), segmentation.cpu(), indices.cpu(), image_id, predicted_probs))
            y_true.extend(indices.cpu().numpy().flatten())
            y_scores.extend(outputs.cpu().numpy().flatten())

    test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = num_correct / total_samples
    avg_pixel_f1 = torch.mean(torch.stack(pixel_f1_scores))
    avg_pixel_accuracy = torch.mean(torch.stack(pixel_accuracies))

    # Calculate precision-recall curve and AUPRC
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recalls, precisions)

    return test_loss, test_accuracy, avg_pixel_f1.item(), avg_pixel_accuracy.item(), confusion_matrix, image_segmentation_pairs, iou_scores, dice_scores, precisions, recalls, specificities, accuracies, rand_errors, variations_of_information, auprc

 # define the function to plot the confusion matrix
def plot_confusion_matrix(confusion_matrix):
    # Define class labels
    classes = ['Negative', 'Positive']

    # Create a custom formatter function to format the values with 2 significant digits
    fmt = lambda x: f"{x:.1e}" if isinstance(x, float) else x

    # Create heatmap
    sns.set(font_scale=1.4)  # Adjust font size
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='.1e', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 16, "ha": 'center', "va": 'center'}, cbar_kws={"shrink": .8})

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    return plt



# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data_directory = 'DATA_DIR' + os.sep + 'SB23_15_SB31_08_handmade_labels_06_10' + os.sep

in_channels = 1  # Number of input channels (RGB image)
out_channels = 1  # Number of output channels
init_features = 32  # Initial number of features


# start evaluating the model
data_directory = os.path.join(DATA_DIR, 'SB23_15_SB31_08_handmade_labels_06_10')
model_dir = '/home/pperritaz/git/deeplogger/output/Bedretto_models'
output_base_dir = 'SB_23_SB31_testing_results'

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Load the dataset
file_IDs = [os.path.join(data_directory, filename) for filename in os.listdir(data_directory)]
complete_dataset = Dataset_np(file_IDs)
complete_dataset = [(image, torch.where(indices == 1, 0, torch.where(indices == 2, 1, torch.where(indices == 3, 1, indices))), image_id) for image, indices, image_id in complete_dataset]
selected_batch_size = len(complete_dataset)
num_images = len(complete_dataset)
test_loader = DataLoader(complete_dataset, batch_size=selected_batch_size, shuffle=False)
batch_size = len(test_loader)


# -------------------This is the code to load and test the model on saved 100 MB borehole images-------------------#
# load the configuration file to extract the paths from the test set :

# file_name = '2D_unet_model07_16BCEloss_all_ATV_adam_batch_size_20_config.p'
# path_to_file = '/home/pperritaz/git/deeplogger/output/Bedretto_models'
# file_path = path_to_file + os.sep + file_name

# Load the file and load the data
# # Open the file and load the data
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
#
#
# test_ids = data["test_set_IDs"]
#
# file_paths = list(test_ids)
# #create a complete dataset by loading the files in the test set
# complete_dataset = Dataset_np(file_paths)
# print(complete_dataset)
#
# selected_batch_size = len(complete_dataset)  # Use the entire dataset for testing
#-----------------------------------------------------------------------------------------------------------------#


defined_treshold = 0.70

results = []

for model_file in os.listdir(model_dir):
    if '2D_unet_model07_16with_almost_allATV_handmade_labels75_epochs' in model_file and model_file.endswith('.pt'):
        model_name = model_file.split('.')[0]
        model_path = os.path.join(model_dir, model_file)

        # Load the model
        model = UNetOTV(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        model.load_state_dict(torch.load(model_path))
        model.to(device).float()

        # Evaluate the model
        criterion = BCELoss()
        test_loss, test_accuracy, avg_pixel_f1, avg_pixel_accuracy, confusion_matrix, image_segmentation_pairs, iou_scores, dice_scores, precisions, recalls, specificities, accuracies, rand_errors, variations_of_information, auprc = evaluate_model(
            model=model, testloader=test_loader, criterion=BCELoss(), device=device, threshold = defined_treshold)

        #print the results
        print( 'for an unseen dataset :' )
        print('Test Accuracy : ', test_accuracy)
        print('Average pixel-wise F1 score : ', avg_pixel_f1)
        print('Average pixel-wise accuracy : ', avg_pixel_accuracy)
        print('Confusion Matrix : ', confusion_matrix)
        print('IoU scores : ', iou_scores)
        print('Dice scores : ', dice_scores)
        print('precision mean : ', np.mean(precisions))
        print('precision std', np.std(precisions))
        print('recall mean : ', np.mean(recalls))
        print('recall std', np.std(recalls))
        print('specificity mean : ', np.mean(specificities))
        print('specificity std', np.std(specificities))
        print('accuracy scores : ', accuracies)
        print('rand errors : ', rand_errors)
        print('variations of information : ', variations_of_information)
        print('AUPRC : ', auprc)
        # Save evaluation metrics
        results.append({
            'model_name': model_name,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'avg_pixel_f1': avg_pixel_f1,
            'avg_pixel_accuracy': avg_pixel_accuracy,
            'confusion_matrix': confusion_matrix,
            'iou_mean': np.mean(iou_scores),
            'iou_std': np.std(iou_scores),
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'median_precisions': np.median(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'auprc': auprc,
            'specificity_mean': np.mean(specificities),
            'specificity_std': np.std(specificities),
        })


        ##plot all images of test dataset against the model and  save them in the created output directory 'plots'
        output_dir = 'plots_SB_Boreholes_final_model_adam_epoch_75'
        os.makedirs(output_dir, exist_ok=True)


        cmap1 = ListedColormap(['none', 'cornflowerblue'])
        cmap2 = ListedColormap(['none', 'black'])

        # create a proxy artist with two legends
        label_patch = mpatches.Patch(color='black', label='Label')
        segmentation_patch = mpatches.Patch(color='lightblue', label='Segmentation')

        # create a legend containing both the mask and the sinusoidal fit
        num_images = len(image_segmentation_pairs)
        for i, (image, segmentation, label, _, predicted_probs) in enumerate(image_segmentation_pairs[:num_images]):
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(2, 3, width_ratios=[0.05, 2, 2], wspace=0.1)

            # Display the image
            ax0 = plt.subplot(gs[0, 1])
            ax0.imshow(image.squeeze().cpu().numpy(),cmap='YlOrBr')
            ax0.set_title('Image')
            ax0.axis('off')

            # Overlay the label using the custom colormap
            ax1 = plt.subplot(gs[0, 2])
            ax1.imshow(image.squeeze().cpu().numpy(), cmap='YlOrBr')
            ax1.imshow(label.squeeze().cpu().numpy(), cmap=cmap2)
            ax1.legend(handles=[label_patch], loc='upper right')

            ax1.set_title('Image with Label')
            ax1.axis('off')

            # Overlay the predicted probabilities
            ax2 = plt.subplot(gs[1, 1])
            ax2.imshow(image.squeeze().cpu().numpy(), cmap='YlOrBr')
            img1 = ax2.imshow(predicted_probs.squeeze(), cmap='nipy_spectral', vmin=0, vmax=1)
            ax2.set_title('Predicted Probabilities \nof Geological Structure')
            ax2.axis('off')

            # Add colorbar to ax[1, 0]
            cbar_ax = plt.subplot(gs[1, 0])
            cbar = fig.colorbar(img1, cax=cbar_ax, orientation='vertical', location='left')

            # Overlay the segmentation mask using the custom colormap
            ax3 = plt.subplot(gs[1, 2])
            ax3.imshow(image.squeeze().cpu().numpy(), cmap='YlOrBr')
            ax3.imshow(segmentation.squeeze().cpu().numpy(), cmap=cmap1)
            ax3.legend(handles=[segmentation_patch], loc='upper right')

            ax3.set_title(f'Generated Segmentation \n(Threshold = {defined_treshold:.2f})')
            ax3.axis('off')

            # Use tight_layout to adjust subplots
            plt.tight_layout()

            # Save the plots on the distant server
            file_path = os.path.join(output_dir, f'plot_{model_name}_pair{i}.png')

            # Check if the file already exists and delete it if it does
            if os.path.exists(file_path):
                os.remove(file_path)

            # Save the plot
            plt.savefig(file_path, dpi=200)
            # Close the plot to free up memory
            plt.close(fig)

        #Here, we want to plot the confusion matrix and save it in the same directory
        plt = plot_confusion_matrix(confusion_matrix)
        file_path = os.path.join(output_dir, f'Confusion_matrix_{model_name}.png')
        # Check if the file already exists and delete it if it does
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(file_path)
        plt.close()

results_df = pd.DataFrame(results)
output_dir_metrics = 'metrics_SB_Boreholes_final_threshold_075'
os.makedirs(output_dir_metrics, exist_ok=True)

results_csv_path = os.path.join(output_dir_metrics, 'evaluation_results_model_final_ATV_model_epoch_75_handmade_labels.csv')
results_df.to_csv(results_csv_path, index=False)

print('Evaluation complete. Results saved to:', results_csv_path)












