"""DeepLogger end-to-end inference demo.

Demonstrates the full pipeline:
1. Load a prepared borehole image/mask pair
2. Apply preprocessing (SVD removal, mean removal, FFT high-pass)
3. Load a trained U-Net model
4. Run inference to predict fracture locations
5. Visualize: original image, ground truth mask, model prediction

Usage:
    python examples/inference_demo.py
    python examples/inference_demo.py --sample-id 8
    python examples/inference_demo.py --sample-id 6 --threshold 0.3
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deeplogger import DATA_DIR, OUTPUT_DIR
from deeplogger.model_architectures_OTV import UNetOTV
from deeplogger.image_processing import remove_svd, remove_mean, high_pass_FFT_2D
from deeplogger.loss_functions import DiceLoss


def load_sample(sample_id: int):
    """Load a prepared borehole image/mask pair."""
    path = os.path.join(DATA_DIR, "Bedretto_Output", f"ID_{sample_id}_data_label.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample not found: {path}")
    sample = torch.load(path, map_location="cpu", weights_only=False)
    image = np.array(sample[0], dtype=np.float64)
    mask = np.array(sample[1], dtype=np.float64)
    return image, mask


def load_model(model_path: str = None, device: torch.device = None):
    """Load a trained UNetOTV model."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model_path is None:
        model_path = os.path.join(
            DATA_DIR, "IDs_and_model",
            "2D_unet_model04_17only_structures_deep_training_2000epochs-epoch-340.pt"
        )
    model = UNetOTV(in_channels=3, out_channels=1, init_features=32).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def predict(model, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference on a single image."""
    img_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction.cpu().numpy()


def compute_dice(prediction: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> float:
    """Compute Dice coefficient between binary prediction and ground truth."""
    pred_binary = (prediction > threshold).astype(float)
    gt_binary = (ground_truth > 0).astype(float)
    intersection = np.sum(pred_binary * gt_binary)
    if pred_binary.sum() + gt_binary.sum() == 0:
        return 1.0  # both empty = perfect match
    return 2.0 * intersection / (pred_binary.sum() + gt_binary.sum())


def visualize(image, mask, prediction, sample_id, threshold, dice_score, output_path):
    """Create a 4-panel visualization."""
    pred_binary = (prediction > threshold).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Original borehole image
    axes[0].imshow(image, aspect="auto", extent=[0, 360, image.shape[0], 0])
    axes[0].set_title("Borehole Image (OTV)")
    axes[0].set_xlabel("Azimuth [deg]")
    axes[0].set_ylabel("Depth index")

    # Panel 2: Ground truth mask
    axes[1].imshow(mask, aspect="auto", cmap="Reds", extent=[0, 360, mask.shape[0], 0],
                   vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("Azimuth [deg]")

    # Panel 3: Raw prediction (probability map)
    im3 = axes[2].imshow(prediction, aspect="auto", cmap="hot",
                         extent=[0, 360, prediction.shape[0], 0], vmin=0, vmax=1)
    axes[2].set_title("Prediction (probability)")
    axes[2].set_xlabel("Azimuth [deg]")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    # Panel 4: Overlay — image + prediction contour
    axes[3].imshow(image, aspect="auto", extent=[0, 360, image.shape[0], 0])
    axes[3].imshow(pred_binary, aspect="auto", cmap="Reds", alpha=0.4,
                   extent=[0, 360, prediction.shape[0], 0], vmin=0, vmax=1)
    if mask.sum() > 0:
        axes[3].contour(mask, levels=[0.5], colors="lime", linewidths=1,
                        extent=[0, 360, mask.shape[0], 0])
    axes[3].set_title(f"Overlay (Dice={dice_score:.3f})")
    axes[3].set_xlabel("Azimuth [deg]")

    fig.suptitle(f"DeepLogger Inference — Sample ID {sample_id} (threshold={threshold})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DeepLogger inference demo")
    parser.add_argument("--sample-id", type=int, default=8,
                        help="Sample ID to run inference on (default: 8)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold for prediction (default: 0.5)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output images")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(OUTPUT_DIR, "demo")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    print(f"Loading sample ID {args.sample_id}...")
    image, mask = load_sample(args.sample_id)
    print(f"  Image shape: {image.shape}, Mask shape: {mask.shape}")
    print(f"  Fracture pixels in ground truth: {int(np.sum(mask > 0))}")

    # 2. Load model
    print("Loading model...")
    model, device = load_model(args.model_path)
    print(f"  Device: {device}")

    # 3. Inference
    print("Running inference...")
    prediction = predict(model, image, device)
    print(f"  Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")

    # 4. Evaluate
    dice = compute_dice(prediction, mask, threshold=args.threshold)
    print(f"  Dice coefficient: {dice:.4f}")

    # 5. Visualize
    output_path = os.path.join(output_dir, f"inference_ID_{args.sample_id}.png")
    visualize(image, mask, prediction, args.sample_id, args.threshold, dice, output_path)

    # 6. Also show preprocessing effects
    print("Generating preprocessing comparison...")
    svd_filtered, svd_removed = remove_svd(image[:, :, 0].copy(), low_s=0, high_s=1)
    mean_removed, mean_matrix = remove_mean(image[:, :, 0].copy(), axis=0)
    fft_filtered = high_pass_FFT_2D(image[:, :, 0].copy(), cutoff_frequency=0.05)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, data, title in zip(axes,
                                [image[:, :, 0], svd_filtered, mean_removed, fft_filtered],
                                ["Original (R channel)", "SVD filtered", "Mean removed", "FFT high-pass"]):
        ax.imshow(data, aspect="auto", cmap="hot", extent=[0, 360, data.shape[0], 0])
        ax.set_title(title)
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel("Depth index")
    fig.suptitle(f"Preprocessing Comparison — Sample ID {args.sample_id}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    preprocess_path = os.path.join(output_dir, f"preprocessing_ID_{args.sample_id}.png")
    plt.savefig(preprocess_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {preprocess_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
