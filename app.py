"""DeepLogger — Streamlit GUI for borehole fracture detection.

Run with:
    streamlit run app.py
"""

import os
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from deeplogger import DATA_DIR, OUTPUT_DIR
from deeplogger.model_architectures_OTV import UNetOTV
from deeplogger.model_architectures_ATV import UNetOTV as UNetATV
from deeplogger.image_processing import remove_svd, remove_mean, high_pass_FFT_2D, high_pass_2D_kernel
from deeplogger.filters import gaussian_blur, neighbor_filter


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

def detect_model_channels(model_path: str) -> int:
    """Detect input channels from a model's saved weights."""
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "encoder1.enc1conv1.weight" in state_dict:
        return state_dict["encoder1.enc1conv1.weight"].shape[1]
    return 3  # default to OTV


@st.cache_resource
def load_model(model_path: str):
    """Load a trained U-Net model (cached across reruns).

    Auto-detects input channels from the saved weights.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_channels = detect_model_channels(model_path)
    if in_channels == 3:
        model = UNetOTV(in_channels=3, out_channels=1, init_features=32)
    else:
        model = UNetATV(in_channels=1, out_channels=1, init_features=32)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device, in_channels


@st.cache_data
def load_sample(path: str):
    """Load a .pt sample file, handling multiple formats.

    Supported formats:
    - [image_3ch, mask]  — OTV with ground truth (360x360x3, 360x360)
    - [image_2d, mask]   — ATV with ground truth (360x360, 360x360)
    - [image_2d]         — ATV data only, no mask (360x360)
    - [image_3ch]        — OTV data only, no mask (360x360x3)

    Returns:
        (image, mask, data_type) where mask may be None and
        data_type is 'otv' or 'atv'.
    """
    sample = torch.load(path, map_location="cpu", weights_only=False)

    # Handle dict (model weights) — shouldn't happen but be safe
    if isinstance(sample, dict):
        return None, None, None

    if not isinstance(sample, (list, tuple)):
        # Single array
        arr = np.array(sample, dtype=np.float64)
        data_type = "otv" if arr.ndim == 3 and arr.shape[2] == 3 else "atv"
        return arr, None, data_type

    image = np.array(sample[0], dtype=np.float64)

    # Determine data type from image shape
    if image.ndim == 3 and image.shape[2] == 3:
        data_type = "otv"
    else:
        data_type = "atv"

    # Extract mask if present
    if len(sample) >= 2:
        candidate = np.array(sample[1], dtype=np.float64)
        # Sanity check: mask should be 2D and same height as image
        if candidate.ndim == 2 and candidate.shape[0] == image.shape[0]:
            mask = candidate
        else:
            mask = None
    else:
        mask = None

    return image, mask, data_type


def find_samples(data_dir: str):
    """Find all directories containing .pt data files (not model weights).

    Peeks at the first file in each directory to check if it's data
    (list/tuple) or model weights (dict). Skips model-weight directories.
    """
    samples = {}
    for subdir in sorted(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        pt_files = sorted(glob.glob(os.path.join(subdir_path, "*.pt")))
        if not pt_files:
            continue
        # Peek at first file to see if it's data or model weights
        try:
            peek = torch.load(pt_files[0], map_location="cpu", weights_only=False)
            if isinstance(peek, dict):
                continue  # model weights, skip
        except Exception:
            continue
        samples[subdir] = pt_files
    return samples


def find_models(search_dirs):
    """Find all .pt model files (state dicts) across search directories."""
    models = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
            try:
                peek = torch.load(f, map_location="cpu", weights_only=False)
                if isinstance(peek, dict):
                    models.append(f)
            except Exception:
                continue
    return models


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def apply_preprocessing(image_channel, settings):
    """Apply selected preprocessing steps to a single 2D channel."""
    result = image_channel.copy()

    if settings.get("svd_enabled"):
        result, _ = remove_svd(result, low_s=0, high_s=settings["svd_components"])

    if settings.get("mean_enabled"):
        result, _ = remove_mean(result, axis=settings["mean_axis"])

    if settings.get("fft_enabled"):
        result = high_pass_FFT_2D(result, cutoff_frequency=settings["fft_cutoff"])

    if settings.get("blur_enabled"):
        result = gaussian_blur(result, kernel_size=settings["blur_kernel"])

    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def prepare_input(image, model_channels):
    """Prepare image for model input, handling channel mismatches.

    If the model expects 3 channels but data is single-channel (ATV),
    stack the channel 3 times to create pseudo-RGB input.
    """
    if image.ndim == 2 and model_channels == 3:
        # ATV data → pseudo-RGB for OTV model
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3 and model_channels == 1:
        # OTV data → grayscale for ATV model (take first channel)
        image = image[:, :, 0]
    return image


def run_inference(model, image, device):
    """Run model inference on an image."""
    img_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction.cpu().numpy()


def compute_dice(prediction, ground_truth, threshold):
    """Compute Dice coefficient."""
    pred_binary = (prediction > threshold).astype(float)
    gt_binary = (ground_truth > 0).astype(float)
    intersection = np.sum(pred_binary * gt_binary)
    total = pred_binary.sum() + gt_binary.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="DeepLogger", page_icon="🔬", layout="wide")
    st.title("DeepLogger — Borehole Fracture Detection")
    st.caption("Deep learning for borehole televiewer data interpretation")

    # --- Sidebar: Data & Model Selection ---
    st.sidebar.header("Data Selection")

    sample_dirs = find_samples(DATA_DIR)
    if not sample_dirs:
        st.error(f"No data found in {DATA_DIR}. Run `python deeplogger/download_data.py` first.")
        return

    dataset = st.sidebar.selectbox("Dataset", list(sample_dirs.keys()))
    pt_files = sample_dirs[dataset]
    sample_names = [os.path.basename(f) for f in pt_files]
    sample_idx = st.sidebar.selectbox("Sample", range(len(sample_names)),
                                       format_func=lambda i: sample_names[i])
    sample_path = pt_files[sample_idx]

    # Find available models
    st.sidebar.header("Model")
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_search_paths = [
        os.path.join(project_root, "models"),
        os.path.join(DATA_DIR, "IDs_and_model"),
        os.path.join(DATA_DIR, "models"),
        os.path.join(OUTPUT_DIR, "Bedretto_models"),
    ]
    available_models = find_models(model_search_paths)
    if not available_models:
        st.sidebar.warning("No trained models found.")
        model_path = None
    else:
        model_names = [os.path.basename(m) for m in available_models]
        model_idx = st.sidebar.selectbox("Model weights", range(len(model_names)),
                                          format_func=lambda i: model_names[i])
        model_path = available_models[model_idx]

    threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.3, 0.05)

    # --- Sidebar: Preprocessing ---
    st.sidebar.header("Preprocessing")

    preprocess = {}
    preprocess["svd_enabled"] = st.sidebar.checkbox("SVD removal", value=False)
    if preprocess["svd_enabled"]:
        preprocess["svd_components"] = st.sidebar.slider("SVD components to remove", 1, 10, 1)

    preprocess["mean_enabled"] = st.sidebar.checkbox("Mean removal", value=False)
    if preprocess["mean_enabled"]:
        preprocess["mean_axis"] = st.sidebar.radio("Mean axis", [0, 1, 2],
                                                     format_func=lambda x: {0: "Column (axis 0)", 1: "Row (axis 1)", 2: "Both"}[x])

    preprocess["fft_enabled"] = st.sidebar.checkbox("FFT high-pass", value=False)
    if preprocess["fft_enabled"]:
        preprocess["fft_cutoff"] = st.sidebar.slider("FFT cutoff", 0.01, 0.3, 0.05, 0.01)

    preprocess["blur_enabled"] = st.sidebar.checkbox("Gaussian blur", value=False)
    if preprocess["blur_enabled"]:
        preprocess["blur_kernel"] = st.sidebar.slider("Blur kernel size", 1, 15, 3, 2)

    # --- Load Data ---
    image, mask, data_type = load_sample(sample_path)

    if image is None:
        st.error("Could not load this file — it may be a model weights file, not data.")
        return

    has_mask = mask is not None
    st.sidebar.info(f"Data type: **{data_type.upper()}** ({'3-ch RGB' if data_type == 'otv' else '1-ch amplitude'})"
                    f"\nMask: {'yes' if has_mask else 'no'}"
                    f"\nShape: {image.shape}")

    # --- Main Layout ---
    if has_mask:
        col_left, col_right = st.columns(2)
    else:
        col_left = st.container()
        col_right = None

    with col_left:
        st.subheader("Borehole Image")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        if data_type == "otv":
            ax1.imshow(image, aspect="auto", extent=[0, 360, image.shape[0], 0])
        else:
            ax1.imshow(image, aspect="auto", cmap="hot", extent=[0, 360, image.shape[0], 0])
        ax1.set_xlabel("Azimuth [deg]")
        ax1.set_ylabel("Depth index")
        st.pyplot(fig1)
        plt.close(fig1)

    if has_mask and col_right is not None:
        n_fracture_pixels = int(np.sum(mask > 0))
        with col_right:
            st.subheader("Ground Truth Mask")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(mask, aspect="auto", cmap="Reds", extent=[0, 360, mask.shape[0], 0],
                       vmin=0, vmax=max(1, mask.max()))
            ax2.set_xlabel("Azimuth [deg]")
            ax2.set_ylabel("Depth index")
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption(f"Fracture pixels: {n_fracture_pixels}")

    # --- Preprocessing Preview ---
    any_preprocessing = any(preprocess.get(k) for k in ["svd_enabled", "mean_enabled", "fft_enabled", "blur_enabled"])
    if any_preprocessing:
        st.subheader("Preprocessing Preview")
        if data_type == "otv":
            original_ch = image[:, :, 0]
            ch_label = "R channel"
        else:
            original_ch = image if image.ndim == 2 else image[:, :, 0]
            ch_label = "Amplitude"
        processed_ch = apply_preprocessing(original_ch, preprocess)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_p1, ax_p1 = plt.subplots(figsize=(6, 5))
            ax_p1.imshow(original_ch, aspect="auto", cmap="hot", extent=[0, 360, original_ch.shape[0], 0])
            ax_p1.set_title(f"Before ({ch_label})")
            ax_p1.set_xlabel("Azimuth [deg]")
            st.pyplot(fig_p1)
            plt.close(fig_p1)
        with col_p2:
            fig_p2, ax_p2 = plt.subplots(figsize=(6, 5))
            ax_p2.imshow(processed_ch, aspect="auto", cmap="hot", extent=[0, 360, processed_ch.shape[0], 0])
            ax_p2.set_title("After preprocessing")
            ax_p2.set_xlabel("Azimuth [deg]")
            st.pyplot(fig_p2)
            plt.close(fig_p2)

    # --- Inference ---
    st.subheader("Model Inference")

    if model_path is None:
        st.warning("No model selected.")
        return

    if st.button("Run Inference", type="primary"):
        with st.spinner("Loading model and running inference..."):
            try:
                model, device, model_channels = load_model(model_path)
            except RuntimeError as e:
                st.error(f"Model loading failed: {e}")
                return

            # Adapt data channels to match model
            model_input = prepare_input(image, model_channels)
            if model_channels != (3 if data_type == "otv" else 1):
                st.info(f"Data is {data_type.upper()} ({'1-ch' if data_type == 'atv' else '3-ch'}) "
                        f"but model expects {model_channels}-ch — adapting automatically.")

            prediction = run_inference(model, model_input, device)

        pred_binary = (prediction > threshold).astype(float)

        # Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        if has_mask:
            dice = compute_dice(prediction, mask, threshold)
            col_m1.metric("Dice Coefficient", f"{dice:.3f}")
        else:
            col_m1.metric("Dice Coefficient", "N/A (no mask)")
        col_m2.metric("Prediction Range", f"[{prediction.min():.3f}, {prediction.max():.3f}]")
        col_m3.metric("Predicted Fracture Pixels", int(pred_binary.sum()))

        # Visualization
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            fig_r1, ax_r1 = plt.subplots(figsize=(6, 6))
            im = ax_r1.imshow(prediction, aspect="auto", cmap="hot",
                              extent=[0, 360, prediction.shape[0], 0], vmin=0, vmax=1)
            ax_r1.set_title("Prediction (probability)")
            ax_r1.set_xlabel("Azimuth [deg]")
            ax_r1.set_ylabel("Depth index")
            plt.colorbar(im, ax=ax_r1, fraction=0.046)
            st.pyplot(fig_r1)
            plt.close(fig_r1)

        with col_r2:
            fig_r2, ax_r2 = plt.subplots(figsize=(6, 6))
            if data_type == "otv":
                ax_r2.imshow(image, aspect="auto", extent=[0, 360, image.shape[0], 0])
            else:
                ax_r2.imshow(image, aspect="auto", cmap="gray", extent=[0, 360, image.shape[0], 0])
            ax_r2.imshow(pred_binary, aspect="auto", cmap="Reds", alpha=0.4,
                         extent=[0, 360, prediction.shape[0], 0], vmin=0, vmax=1)
            if has_mask and mask.sum() > 0:
                ax_r2.contour(np.flipud(mask), levels=[0.5], colors="lime", linewidths=1.5,
                              extent=[0, 360, mask.shape[0], 0])
            ax_r2.set_title(f"Overlay (threshold={threshold})")
            ax_r2.set_xlabel("Azimuth [deg]")
            ax_r2.set_ylabel("Depth index")
            st.pyplot(fig_r2)
            plt.close(fig_r2)


if __name__ == "__main__":
    main()
