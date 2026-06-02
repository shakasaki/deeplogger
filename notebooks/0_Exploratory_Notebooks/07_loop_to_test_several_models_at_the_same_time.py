# Import the necessary libraries
import os
import numpy as np
import torch
from deeplogger.model_architectures_ATV import UNetOTV
from deeplogger.dataloader import Dataset_np as Dataset
from torch.utils.data import DataLoader
from deeplogger import DATA_DIR, OUTPUT_DIR
from torch.nn import BCELoss
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, recall_score, adjusted_rand_score, mutual_info_score, precision_recall_curve, auc

# Paths and parameters
data_directory = os.path.join(DATA_DIR, 'SB23_15_SB31_08_handmade_labels_06_10')
model_dir = '/home/pperritaz/git/deeplogger/output/Bedretto_models'
output_base_dir = 'SB_23_SB31_testing_results'

# Model parameters
in_channels = 1
out_channels = 1
init_features = 32

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Load the dataset
file_IDs = [os.path.join(data_directory, filename) for filename in os.listdir(data_directory)]
complete_dataset = Dataset(file_IDs)
complete_dataset = [(image, torch.where(indices == 1, 0, torch.where(indices == 2, 1, torch.where(indices == 3, 1, indices)))) for image, indices in complete_dataset]
batch_size = len(complete_dataset)
num_images = len(complete_dataset)
test_loader = DataLoader(complete_dataset, batch_size=batch_size, shuffle=True)
batch_size = len(test_loader)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
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
        for images, indices in test_loader:
            images = images.to(device).float()
            indices = indices.unsqueeze(1).to(device).float()

            outputs = model(images)
            if len(outputs.shape) == 2:
                outputs = outputs.unsqueeze(1)

            indices = indices.view_as(outputs)
            loss = criterion(outputs, indices)
            test_losses.append(loss.item())

            predicted = torch.round(outputs)
            num_correct += (predicted == indices).sum().item()
            total_samples += indices.numel()

            segmentation = predicted.byte()
            indices = indices.byte()

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

            intersection = (segmentation & indices).sum().float()
            union = (segmentation | indices).sum().float()
            pixel_f1 = (2.0 * intersection + smooth) / (segmentation.sum() + indices.sum() + smooth)
            pixel_f1_scores.append(pixel_f1)

            #allow a command to avoid division by zero
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
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            correct_pixels = (segmentation == indices).sum().float()
            total_pixels = segmentation.numel()
            pixel_accuracy = correct_pixels / total_pixels
            pixel_accuracies.append(pixel_accuracy)


            precisions.append(precision)
            recalls.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)

            rand_errors.append(adjusted_rand_score(indices.cpu().numpy().flatten(), predicted.cpu().numpy().flatten()))
            variations_of_information.append(mutual_info_score(indices.cpu().numpy().flatten(), predicted.cpu().numpy().flatten()))

            image_segmentation_pairs.extend(zip(images.cpu(), segmentation.cpu(), indices.cpu()))

            y_true.extend(indices.cpu().numpy().flatten())
            y_scores.extend(outputs.cpu().numpy().flatten())

    test_loss = sum(test_losses) / len(test_losses)
    test_accuracy = num_correct / total_samples
    avg_pixel_f1 = torch.mean(torch.stack(pixel_f1_scores))
    avg_pixel_accuracy = torch.mean(torch.stack(pixel_accuracies))

    # Calculate precision-recall curve and AUPRC
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    return test_loss, test_accuracy, avg_pixel_f1.item(), accuracy, precisions, recalls, specificities, accuracies, confusion_matrix, image_segmentation_pairs, iou_scores, dice_scores, rand_errors, variations_of_information, auprc

# Function to plot the confusion matrix
def plot_confusion_matrix(confusion_matrix):
    classes = ['Negative', 'Positive']
    sns.set(font_scale=1.4)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='.1e', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 16, "ha": 'center', "va": 'center'}, cbar_kws={"shrink": .8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    return plt

# Function to plot the precision-recall curve
def plot_precision_recall_curve(precision, recall):
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    return plt

# Loop through models and evaluate
results = []
for model_file in os.listdir(model_dir):
    if 'ATV' in model_file and model_file.endswith('.pt'):
        model_name = model_file.split('.')[0]
        model_path = os.path.join(model_dir, model_file)

        # Load the model
        model = UNetOTV(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        model.load_state_dict(torch.load(model_path))
        model.to(device).float()

        # Evaluate the model
        criterion = BCELoss()
        test_loss, test_accuracy, avg_pixel_f1, accuracy, precisions, recalls, specificities, accuracies, confusion_matrix, image_segmentation_pairs, iou_scores, dice_scores, rand_errors, variations_of_information, auprc = evaluate_model(model, test_loader, criterion, device)

        # Save evaluation metrics
        results.append({
            'model_name': model_name,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'avg_pixel_f1': avg_pixel_f1,
            'avg_pixel_accuracy': accuracy,
            'iou_mean': np.mean(iou_scores),
            'iou_median': np.median(iou_scores),
            'iou_std': np.std(iou_scores),
            'dice_mean': np.mean(dice_scores),
            'dice_median': np.median(dice_scores),
            'dice_std': np.std(dice_scores),
            'precision_mean': np.mean(precisions),
            'precision_median': np.median(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_median': np.median(recalls),
            'recall_std': np.std(recalls),
            'rand_error_mean': np.mean(rand_errors),
            'rand_error_median': np.median(rand_errors),
            'rand_error_std': np.std(rand_errors),
            'vi_mean': np.mean(variations_of_information),
            'vi_median': np.median(variations_of_information),
            'vi_std': np.std(variations_of_information),
            'auprc': auprc,
            'specificity_mean': np.mean(specificities),
            'specificity_median': np.median(specificities),
            'specificity_std': np.std(specificities),
        })

        # Create directory for model output
        model_output_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Plot and save images
        cmap1 = ListedColormap(['none', 'red'])
        cmap2 = ListedColormap(['none', 'orange'])
        for i, (image, segmentation, label) in enumerate(image_segmentation_pairs[:num_images]):
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            ax[0].imshow(image.squeeze().cpu().numpy(), cmap='viridis')
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            ax[1].imshow(image.squeeze().cpu().numpy(), cmap='viridis')
            ax[1].imshow(label.squeeze().cpu().numpy(), cmap=cmap2, alpha=0.7)
            ax[1].set_title('Original Label')
            ax[1].axis('off')
            ax[2].imshow(image.squeeze().cpu().numpy(), cmap='viridis')
            ax[2].imshow(segmentation.squeeze().cpu().numpy(), cmap=cmap1, alpha=0.7)
            ax[2].set_title('Generated Segmentation Mask')
            ax[2].axis('off')

            file_path = os.path.join(model_output_dir, f'plot_pair_{i}.png')
            if os.path.exists(file_path):
                os.remove(file_path)
            plt.savefig(file_path)
            plt.close(fig)

        # Plot and save confusion matrix
        plt = plot_confusion_matrix(confusion_matrix)
        file_path = os.path.join(model_output_dir, f'Confusion_matrix.png')
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(file_path)
        plt.close()

        # Plot and save precision-recall curve
        plt = plot_precision_recall_curve(precisions, recalls)
        file_path = os.path.join(model_output_dir, f'Precision_Recall_Curve.png')
        if os.path.exists(file_path):
            os.remove(file_path)
        plt.savefig(file_path)
        plt.close()

# Save results to CSV
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(output_base_dir, 'evaluation_results.csv')
results_df.to_csv(results_csv_path, index=False)

print('Evaluation complete. Results saved to:', results_csv_path)
