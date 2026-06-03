# DeepLogger — Technical Design Document

**Project:** Automated fracture picking in borehole televiewer images  
**Institution:** ETH Zürich, D-ERDW  
**Context:** VALTER / Bedretto Underground Laboratory  
**Status:** Research prototype (v2 architectures, full GUI pipeline)

---

## Contents

1. [Problem & Data](#1-problem--data)
2. [Preprocessing](#2-preprocessing)
3. [Model Architectures](#3-model-architectures)
4. [Loss Functions](#4-loss-functions)
5. [Training Strategy](#5-training-strategy)
6. [Inference & GUI Pipeline](#6-inference--gui-pipeline)
7. [Fine-tuning Loop](#7-fine-tuning-loop)
8. [Reproducing Training on a GPU Server](#8-reproducing-training-on-a-gpu-server)
9. [References](#9-references)

---

## 1. Problem & Data

### 1.1 Borehole Televiewer Imaging

Borehole televiewer logging produces continuous 360° panoramic images of borehole walls from which geological structures (fractures, bedding planes, fault zones) are characterised by their depth, dip, and azimuth. Two instrument types are in use at the Bedretto Lab:

- **ATV (Acoustic Televiewer):** Emits ultrasonic pulses; records reflected amplitude and travel time as two single-channel float images. Amplitude ≈ wall reflectivity; travel time ≈ borehole radius. Raw values are in arbitrary instrument units. One input channel.
- **OTV (Optical Televiewer):** Records a 3-channel RGB panoramic photo of the borehole wall. Values in [0, 255]. Three input channels.

In both cases the image is unrolled azimuthally, producing a 2D array of shape `(depth_rows, azimuth_cols)` where `azimuth_cols ≈ 360`. Geological structures appear as sinusoidal curves because a planar fracture intersects a cylinder in an ellipse, which projects to a sine when unrolled [Gaillot et al., 2007].

The task is binary semantic segmentation: classify each pixel as fracture (foreground) or background. Foreground pixels are typically <1% of the image, producing extreme class imbalance [Li et al., 2019].

### 1.2 Dataset

Training data for the Perritaz (2024) thesis was collected at the Bedretto Underground Laboratory (BedrettoLab), Ticino, Switzerland [Ma et al., 2021; Plenkers et al., 2023]. Sixteen boreholes were logged; ATV and OTV data were acquired with ALT ABI40/OBI40 instruments [Castilla et al., 2022].

Labels were created by manual interpretation in WellCAD™ followed by rasterization of fracture sine parameters onto the image grid. Label encoding: value 1 = background, value 2 = fracture (primary), value 3 = fracture (secondary). During training, values 2 and 3 are mapped to 1 (foreground) and value 1 is mapped to 0 (`_binarize_mask` in `train.py`).

- **ATV training set:** 1709 `[image, mask]` snippets, SVD-filtered amplitude (k=3, see §2.3), float32  
  Path: `~/DATA/Bedretto varia/Data_Msc_Thesis_Perritaz/data/Training_data_manually_Drawn_labels/atv_data_label/`
- **OTV training set:** Similar structure, 3-channel RGB  
- **Test boreholes:** SB 2.3 and SB 3.1 were held out for evaluation [Perritaz, 2024]

### 1.3 Data Format

Training snippets are stored as `.pt` files containing a list `[image_tensor, mask_tensor]`:
- ATV: image shape `(1, H, W)`, dtype float32; mask shape `(1, H, W)`, dtype int
- OTV: image shape `(3, H, W)`, dtype float32 (uint8 RGB normalised to [0,1]); mask same

The `BoreholeDataset` class (`deeplogger/dataloader.py`) loads these via `torch.load` and exposes them through the standard PyTorch `Dataset` interface.

---

## 2. Preprocessing

### 2.1 LAS File Ingestion

Raw data arrives as LAS files. `deeplogger/las_reader.py` parses LAS headers (STRT/STOP/STEP/null/delimiter) and data using `polars` for fast columnar I/O. ATV data is loaded as float32 with NULL values → NaN; OTV data is loaded as uint8 RGB. The `BoreholeLog` container supports windowed reads (depth range slicing) and conversion to Zarr for caching.

### 2.2 Multiscale Pyramid

`deeplogger/pyramid.py` builds a multiscale Zarr pyramid via depth-axis block mean downsampling (×2 per level). This allows the GUI viewer to serve the appropriate resolution for the current zoom level without loading the full log.

### 2.3 SVD Destriping

Borehole logs frequently contain depth-correlated stripes (tool rotation, electronic drift). SVD destriping removes the top-*k* singular components from the image matrix:

```
X = U Σ Vᵀ  →  X_destriped = X - Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ
```

Implemented in `deeplogger/image_processing.py:remove_svd_components`. The GUI viewer exposes a spinbox to set *k* interactively with live re-pyramiding.

**Critical for training consistency — ATV only.** The ATV training snippets in `atv_data_label/` were created *after* applying SVD filtering with k = 3 (Perritaz, 2024, Section 4.3.2, p. 14 of the thesis PDF):

> "The acoustic data was pre-processed by applying a singular value decomposition (SVD) filter, removing the largest three singular values. Prevalent noise traces could be withdrawn, resulting in clearer data images."

This means the model was trained on SVD-filtered data. For consistent inference on new ATV logs, **apply SVD with k = 3 before passing the image to the model** — otherwise the input distribution differs from training data and predictions degrade. OTV data was not pre-processed with SVD.

### 2.4 Input Normalisation (v2 models)

Raw ATV amplitude values vary significantly across boreholes and logging runs. UNetATV and AttentionUNetATV apply per-sample min-max normalisation inside `forward()`:

```
x̂ = (x − min(x)) / (max(x) − min(x) + ε)
```

This is implemented in `_norm_input` (`model_architectures_v2.py`) and applied to each sample in the batch independently, making the model invariant to global amplitude offsets between acquisition runs.

---

## 3. Model Architectures

See also: `docs/architecture_diagram.png` (generated by `scripts/plot_architectures.py`).

### 3.1 UNetATV v1 (Perritaz baseline)

**File:** `deeplogger/model_architectures_ATV.py`  
**Class:** `UNetOTV` (imported as `UNetATV`)

Faithful reproduction of the architecture from Perritaz (2024), which is itself adapted from the original 2D U-Net [Ronneberger et al., 2015]:

| Stage | Operation | Output channels |
|---|---|---|
| Encoder 1–3 | 2× Conv3×3 + BN + ReLU, then MaxPool2d(k=2, s=2) | 32 → 64 → 128 |
| Encoder 4 | 2× Conv3×3 + BN + ReLU, then MaxPool2d(**k=3, s=1, p=1**) | 256 |
| Bottleneck | 2× Conv3×3 + BN + ReLU | 512 |
| Decoder 4–1 | ConvTranspose2d(k=2, s=2) + cat(skip) + 2× Conv3×3 + BN + ReLU | 256 → 128 → 64 → 32 |
| Head | Conv1×1 + Sigmoid | 1 |

Pool4 uses `kernel_size=3, stride=1, padding=1`, which is size-preserving (a no-op for spatial dimensions). This was a design choice in the thesis to give the bottleneck a wider receptive field without spatial downsampling at level 4. Skips 3 and 4 require bilinear interpolation to match decoder upsampling; skips 1 and 2 require `H, W % 4 == 0`.

**Parameters:** ~7.76M

### 3.2 UNetATV v2 (Perritaz + normalisation)

**File:** `deeplogger/model_architectures_v2.py`  
**Class:** `UNetATV`

Identical architecture to v1, with two changes:
1. Per-sample min-max normalisation applied at the start of `forward()` (Section 2.4).
2. All four skip connections use `F.interpolate` before the decoder `cat`, so any `(H, W)` input is accepted without alignment constraints.

`pool4` is still `MaxPool2d(k=3, s=1, p=1)` (size-preserving), matching the thesis exactly.

**Parameters:** ~7.76M  
**Default model type:** `ModelType.UNET_ATV_V2`

### 3.3 AttentionUNetATV

**File:** `deeplogger/model_architectures_v2.py`  
**Class:** `AttentionUNetATV`

Extended architecture addressing two weaknesses of v1/v2: (a) the model cannot suppress irrelevant background features in skip connections, and (b) overfitting on the small dataset.

**Attention gates** [Oktay et al., 2018]: At each decoder level, before concatenating the encoder skip with the upsampled decoder feature, an `AttentionGate` computes a per-spatial-location weight map α ∈ (0, 1):

```
g_up = interpolate(g, size=x.shape[2:])          # align decoder to encoder
α    = σ(ψ(ReLU(W_g(g_up) + W_x(x))))            # 1×1 conv + sigmoid
output = x ⊙ α                                    # element-wise gate
```

where `g` is the decoder (gating) signal and `x` is the encoder skip feature. This suppresses background responses before they enter the decoder, directly addressing the extreme foreground/background imbalance.

**Spatial Dropout2d**: `Dropout2d` drops entire feature maps (channels) during training, which prevents co-adaptation between feature detectors [Srivastava et al., 2014]. Applied to:
- Encoders 3 and 4: p = 0.1
- Bottleneck: p = 0.2

**Pool4 correction**: The size-preserving pool4 from v1/v2 is replaced with real `MaxPool2d(k=2, s=2)`, giving a clean 4-level encoder where the bottleneck is at `H/16, W/16`. This removes the spatial quirk and simplifies the skip connection alignment.

**Parameters:** ~7.89M

| Feature | UNetATV v1 | UNetATV v2 | AttentionUNetATV |
|---|---|---|---|
| Input normalisation | ✗ | ✓ | ✓ |
| Attention gates | ✗ | ✗ | ✓ |
| Spatial Dropout2d | ✗ | ✗ | ✓ |
| Pool4 stride | 1 (no-op) | 1 (no-op) | 2 (real) |
| Bottleneck depth | H/8 | H/8 | H/16 |
| Any (H, W) input | ✗ | ✓ | ✓ |

### 3.4 UNetOTV (3-channel OTV)

**File:** `deeplogger/model_architectures_OTV.py`  
**Class:** `UNetOTV`

Same U-Net topology as v1, with `in_channels=3` for RGB OTV data. Pool4 is `MaxPool2d(k=3, s=2, no padding)` — this is a genuine stride-2 downsampling, requiring `H % 16 == 8` and `W % 16 == 8` for skip connections to align correctly. The inference pipeline pads both dimensions to satisfy this constraint before calling the model.

**Architecture dispatch** is controlled by `ModelType` enum in `config.py` and resolved in `train.py:_build_model()`.

---

## 4. Loss Functions

All loss functions are in `deeplogger/loss_functions.py`.

### 4.1 Binary Cross-Entropy (BCE)

Standard pixel-wise BCE loss (`torch.nn.BCELoss`). Penalises per-pixel errors equally regardless of class frequency. Adequate as a baseline but sensitive to class imbalance: with <1% foreground, it is easy to achieve low BCE by predicting all-background.

```
BCE(Y, Ŷ) = −[Y log(Ŷ) + (1−Y) log(1−Ŷ)]
```

Used in Perritaz (2024) for ATV training (Adam, lr = 0.1).

### 4.2 Dice Loss

Directly optimises the Dice Similarity Coefficient (DSC) between prediction and ground truth [Sudre et al., 2017]:

```
DL(Y, Ŷ) = 1 − (2·ΣYŶ + ε) / (ΣY + ΣŶ + ε)
```

where ε = 1 (Laplace smoothing) avoids division by zero in the all-zero case. The sum is over all pixels in the batch. Dice loss is robust to class imbalance because it normalises by the total predicted and ground-truth area, not per-pixel count. Used in Perritaz (2024) for OTV training.

### 4.3 BCE+Dice Combined Loss

The preferred loss for training on imbalanced data [Alexakis & Armenakis, 2020; Sudre et al., 2017]:

```
L(Y, Ŷ) = w_bce · BCE(Y, Ŷ) + w_dice · Dice(Y, Ŷ)
```

Default weights: `w_bce = w_dice = 0.5`. BCE stabilises early training gradients; Dice forces the model to improve foreground recall. The implementation (`BCEDiceLoss`) is shape-agnostic — it accepts `(B, H, W)` or `(B, 1, H, W)` predictions by flattening to 1D before the Dice computation.

Recommended hyperparameters from Perritaz (2024): Adam optimizer, lr = 5×10⁻⁴.

### 4.4 Smooth F1 Loss

Differentiable approximation of the F1 score: `−TP / (TP + ½(FP + FN) + ε)`. Implemented as `smoothf1_loss`. Negative because minimisation is assumed. Less used in practice; BCE+Dice is preferred.

---

## 5. Training Strategy

### 5.1 Configuration

All training hyperparameters are captured in `TrainingConfig` (`deeplogger/config.py`):

| Parameter | Default | Notes |
|---|---|---|
| `model_type` | `UNET_ATV_V2` | Architecture selector |
| `loss_type` | `BCE` | Recommend `BCE_DICE` for new ATV training |
| `optimizer_type` | `ADAM` | Adam recommended for BCE+Dice [Perritaz, 2024] |
| `learning_rate` | 0.001 | Use 5×10⁻⁴ for BCE+Dice |
| `max_epochs` | 600 | Thesis best: epoch 75 (ATV+BCE+Adam) |
| `batch_size` | 20 | |
| `lr_step_size` | 1 | StepLR applied every epoch |
| `lr_gamma` | 0.75 | Multiplicative decay |
| `validate_every` | 15 | Best model checkpoint saved on val loss |
| `seed` | 100 | Reproducibility |
| `augment` | True | Random horizontal flips |

### 5.2 Dataset Split

Current split is random: 10% test, 20% of remainder for validation (`test_fraction`, `val_fraction`). **Known issue:** random splitting of spatially correlated borehole snippets causes data leakage — snippets from the same depth range appear in both train and test sets. The correct approach is borehole-stratified splitting (whole boreholes SB2.3 and SB3.1 held out as test, all others for train/val). This is a pending TODO.

### 5.3 Augmentation

Random horizontal flip (`p=0.5`) applied jointly to image and mask. Horizontal flip corresponds to a 180° rotation of the borehole viewing direction, which is a valid physical transformation. Vertical flip is not used (would reverse depth ordering).

### 5.4 Optimizers

- **Adam** (`OptimizerType.ADAM`): Adaptive learning rate; recommended for BCE+Dice loss. Converges faster than SGDM — Perritaz (2024) found that Adam reached optimal ATV performance at epoch 75, whereas SGDM required more epochs for equivalent sensitivity.
- **SGD with momentum** (`OptimizerType.SGD`): Momentum = 0.9. Used in thesis for OTV training with Dice and BCE losses, lr = 0.1 [Perritaz, 2024, citing Fu et al., 2023].

### 5.5 Thesis Best Results (ATV)

From Perritaz (2024), Table 4 — best ATV model (BCE loss, Adam, epoch 75, threshold 0.5):
- Accuracy: 99%
- Sensitivity: 81–90% (mean ± std across test boreholes)
- Specificity: ~100% (dominated by background class)
- Precision: 2% ± 3–9% (expected under extreme imbalance [Li et al., 2019])
- F1 score: 13–32%

The low precision and F1 reflect the <1% foreground fraction: even high sensitivity yields low precision because most predicted positives are false. Post-processing (sinusoidal fitting, connected component filtering) is essential for geometric characterization downstream.

**Note:** The original thesis ATV model weights (Adam, epoch 75) were never committed to the repository. The `models/` directory contains only OTV-trained weights (detectable by `encoder1.enc1conv1.weight.shape[1] == 3`). ATV retraining is required before inference on ATV data is useful.

---

## 6. Inference & GUI Pipeline

### 6.1 Full Pipeline

```
LAS file
  → las_reader.py:read_las_data()       # polars, windowed read
  → BoreholeLog.to_zarr()               # cache to Zarr
  → pyramid.py:build_depth_pyramid()    # multiscale ×2 block mean
  → gui/viewer.py:LogViewer             # pyqtgraph live browse
        ↓ (optional SVD destripe, re-pyramid)
        ↓ "Label window…" button
  → gui/labeler.py:launch_labeler()     # napari, ≤20k rows
        ↓ (paint mask or pick sinusoid)
  → saves [image, mask].pt + picks.csv
        ↓
  → gui/inferencer.py:run_predict()     # model forward pass
  → probability map → binary overlay (threshold slider)
        ↓ human correction (edit overlay layer in napari)
  → saves correction [image, mask].pt
        ↓
  → train.py:finetune()                 # background thread
```

### 6.2 GUI Entry Points

- **Launcher** (`deeplogger/gui/launcher.py`): Qt start screen with three options: Browse/Label, Inspect Bundles, Inference+Correct. Entry point: `python -m deeplogger.gui.viewer` or double-click `deeplogger.desktop`.
- **Log Viewer** (`deeplogger/gui/viewer.py:LogViewer`): Fast pyqtgraph browse, depth-locked Y-axis, pyramid-level swapping on scroll/zoom, SVD destripe spinbox, colormap selector, "Label window…" handoff to napari.
- **Bundle Inspector** (`deeplogger/gui/bundle_inspector.py:BundleInspector`): Browse directories of `[image, mask]` `.pt` files. Left panel: image; right panel: mask overlay. Keyboard ← → navigation, mask coverage % in status bar. Handles ATV `(H,W)` and OTV `(H,W,3)`.
- **Labeler** (`deeplogger/gui/labeler.py`): napari session for a single depth window. Supports two modes: mask paint (Labels layer) and sinusoid pick (interactive mouse: click = depth, drag x = azimuth/phase, drag y = dip/amplitude). Saves `[image, mask].pt` and appends to `picks.csv`.
- **Inferencer** (`deeplogger/gui/inferencer.py`): napari session with model, inference, and correction panels. Loads a `.pt` model, runs inference on the visible window, displays probability map and binary overlay. Correction panel: user edits overlay, clicks "Save Correction" to write a bundle, or "Fine-tune Model" to trigger a background fine-tune.

### 6.3 Padding Constraint

The OTV model's `pool4(stride=2, kernel=3, no-padding)` introduces a non-trivial constraint: the output spatial size is `floor((H - 3) / 2) + 1`. For the ConvTranspose2d in the decoder to produce an output matching the skip connection, both `H` and `W` must satisfy `dim % 16 == 8`. The inference pipeline (`run_predict` in `inferencer.py`) pads both H and W to the next valid size using edge padding, then crops the output back to the original size:

```python
H_pad = _valid_height(H_orig)   # next integer satisfying %16==8
W_pad = _valid_height(W_orig)
img   = np.pad(img, [(0, H_pad-H_orig), (0, W_pad-W_orig), ...], mode="edge")
pred  = model(tensor)
pred  = pred[:H_orig, :W_orig]  # crop to original size
```

The ATV models (v1/v2/attention) do not have this constraint — pool4 is either size-preserving or stride=2 with symmetric architecture, and all skips use `F.interpolate`.

---

## 7. Fine-tuning Loop

After reviewing inference output on a new borehole, the user can correct the binary overlay directly in napari (add/erase fracture pixels). The corrected mask is saved as a `[image, mask]` `.pt` bundle via "Save Correction". These bundles accumulate in an output directory.

`train.py:finetune(model_path, data_dir, n_epochs, lr)`:
1. Detects ATV vs OTV from the state dict (`encoder1.enc1conv1.weight.shape[1]`).
2. Builds the appropriate model class and loads the saved weights.
3. Trains for `n_epochs` epochs on all `.pt` files in `data_dir` using BCE loss and Adam (lr default 1×10⁻⁴).
4. Saves the updated state dict as `<stem>_finetuned.pt` in the same directory.
5. Reports progress via an optional `progress_callback(epoch, n_epochs, loss)`.

The fine-tune runs in a background `QThread` in the inferencer GUI so the napari window remains responsive. The small learning rate (1×10⁻⁴, ~10× smaller than initial training) prevents catastrophic forgetting of the original weights.

---

## 8. Reproducing Training on a GPU Server

### 8.1 Data Preparation

The ATV training data must be SVD-filtered before use. The `.pt` snippets in `atv_data_label/` were
already filtered with k = 3 by Perritaz (2024) (Section 4.3.2, PDF p. 14). If you create new training
snippets from raw LAS files, apply SVD (k=3) first using `deeplogger/image_processing.py:remove_svd_components`.

OTV training snippets require **no preprocessing** — the thesis trained directly on raw RGB values.

Amplitude normalization is **not** required as a data preparation step for either type: the v2 ATV models
(`UNetATV`, `AttentionUNetATV`) apply per-sample min-max normalisation inside `forward()` automatically.

### 8.2 Environment Setup

```bash
# 1. Clone
git clone https://github.com/shakasaki/deeplogger.git
cd deeplogger

# 2. Create conda environment (Python 3.12)
conda create -n deeplogger python=3.12 -y
conda activate deeplogger

# 3. Install CUDA-enabled PyTorch FIRST (before the package install).
#    Check your server's CUDA version with: nvidia-smi
#    Replace cu124 with cu118 or cu121 to match your CUDA version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install DeepLogger and dev dependencies
pip install -e ".[dev]"

# 5. Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** `environment.yml` at the repo root does not pin a CUDA torch build — it resolves
> the CPU wheel by default. Always install torch manually with the CUDA wheel before running
> `pip install -e .` on a GPU server.

### 8.3 Training Command

Train both v2 ATV architectures in sequence and compare:

```bash
python scripts/run_training.py \
    --data-dir ~/DATA/Bedretto\ varia/Data_Msc_Thesis_Perritaz/data/Training_data_manually_Drawn_labels/atv_data_label/ \
    --model-dir models/ \
    --data-type atv \
    --models unet_atv_v2 attention_unet_atv \
    --loss bce_dice \
    --optimizer adam \
    --lr 5e-4 \
    --epochs 200 \
    --run-name atv_bce_dice_adam
```

Outputs written to `models/`:
- `atv_bce_dice_adam_unet_atv_v2_best.pt` — best checkpoint by validation loss
- `atv_bce_dice_adam_attention_unet_atv_best.pt`
- `atv_bce_dice_adam_report.md` — training summary + test evaluation metrics

To reproduce the thesis baseline (BCE, Adam, 200 epochs):

```bash
python scripts/run_training.py \
    --data-dir <atv_data_label_path> \
    --model-dir models/ \
    --data-type atv \
    --models unet_atv_v1 \
    --loss bce \
    --optimizer adam \
    --lr 1e-1 \
    --epochs 200 \
    --run-name atv_bce_adam_baseline
```

### 8.4 Expected Runtime

On a single GPU (e.g. NVIDIA A100 / RTX 3090), 1709 samples, batch size 20, 200 epochs:
approximately 30–60 minutes per model depending on GPU. The script logs epoch losses to stdout
and saves the best checkpoint whenever validation loss improves (validated every 15 epochs by default).

### 8.5 Copying Results Back

After training, copy the weights and report to your development machine:

```bash
scp gpu-server:~/deeplogger/models/atv_bce_dice_adam_*.pt  models/
scp gpu-server:~/deeplogger/models/atv_bce_dice_adam_report.md  docs/
```

---

## 9. References

[Achtziger-Zupančič et al., 2024] Achtziger-Zupančič, P., Ceccato, A., Zappone, A. S., Pozzi, G., Shakas, A., Amann, F., Behr, W. M., Escallon Botero, D., Giardini, D., Hertrich, M., et al. (2024). Selection and characterisation of the target fault for fluid-induced activation and earthquake rupture experiments. *EGUsphere*, 2024, 1–38.

[Alexakis & Armenakis, 2020] Alexakis, E. and Armenakis, C. (2020). Evaluation of UNet and UNet++ architectures in high resolution image change detection applications. *ISPRS — The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, 43, 1507–1514. https://doi.org/10.3390/rs12101672

[Assous et al., 2014] Assous, S., Elkington, P., Clark, S., and Whetton, J. (2014). Automated detection of planar geologic features in borehole images. *Geophysics*, 79(1), D11–D19.

[Bröker et al., 2024] Bröker, K., Ma, X., Zhang, S., Doonechaly, N. G., Hertrich, M., Klee, G., Greenwood, A., Caspari, E., and Giardini, D. (2024). Constraining the stress field and its variability at the BedrettoLab: Elaborated hydraulic fracture trace analysis. *International Journal of Rock Mechanics and Mining Sciences*, 178, 105739.

[Castilla et al., 2022] Castilla, R., Serbeto, F., Christe, F., Meier, P., Bethmann, F., Alcolea, A., Dyer, B., Hertrich, M., and Ma, X. (2022). Data integration and model updating in a multi-stage stimulation in the Bedretto Lab, Switzerland. In *ARMA US Rock Mechanics/Geomechanics Symposium*, ARMA–2022.

[Dias et al., 2020] Dias, L. O., Bom, C. R., Faria, E. L., Valentín, M. B., Correia, M. D., de Albuquerque, M. P., de Albuquerque, M. P., and Coelho, J. M. (2020). Automatic detection of fractures and breakout patterns in acoustic borehole image logs using fast-region convolutional neural networks. *Journal of Petroleum Science and Engineering*, 191, 107099.

[Fu et al., 2023] Fu, J., Wang, B., Zhang, H., Zhang, Z., Chen, W., and Zheng, N. (2023). When and why momentum accelerates SGD: An empirical study. arXiv:2306.09000.

[Gaillot et al., 2007] Gaillot, P., Brewer, T., Pezard, P., and Yeh, E. (2007). Borehole imaging tools — principles and applications. *Scientific Drilling*, 5.

[Grossi & Buscema, 2007] Grossi, E. and Buscema, M. (2007). Introduction to artificial neural networks. *European Journal of Gastroenterology & Hepatology*, 19(12), 1046–1054.

[Han et al., 2021] Han, X., Zhang, Z., Ding, N., Gu, Y., Liu, X., Huo, Y., Qiu, J., Yao, Y., Zhang, A., Zhang, L., et al. (2021). Pre-trained models: Past, present and future. *AI Open*, 2, 225–250.

[Han et al., 2023] Han, S., Xiao, X., Song, B., Guan, T., Zhang, Y., and Lyu, M. (2023). Automatic borehole fracture detection and characterization with tailored Faster R-CNN and simplified Hough transform. *Engineering Applications of Artificial Intelligence*, 126, 107024.

[Janthakal & Hosalli, 2021] Janthakal, S. and Hosalli, G. (2021). A binary cross entropy U-Net based lesion segmentation of granular parakeratosis. In *2021 International Conference on Advancements in Electrical, Electronics, Communication, Computing and Automation (ICAECA)*, pages 1–7. IEEE.

[Li et al., 2019] Li, Z., Kamnitsas, K., and Glocker, B. (2019). Overfitting of neural nets under class imbalance: Analysis and improvements for segmentation. In *MICCAI 2019*, Lecture Notes in Computer Science, pages 402–410. Springer.

[Ma et al., 2021] Ma, X., Hertrich, M., Amann, F., Bröker, K., Gholizadeh Doonechaly, N., Gischig, V., Hochreutener, R., Kästli, P., Krietsch, H., Marti, M., et al. (2021). Multi-disciplinary characterizations of the BedrettoLab — a unique underground geoscience research facility. *Solid Earth Discussions*, 2021, 1–40.

[Moccia et al., 2018] Moccia, S., De Momi, E., El Hadji, S., and Mattos, L. S. (2018). Blood vessel segmentation algorithms — review of methods, datasets and evaluation metrics. *Computer Methods and Programs in Biomedicine*, 158, 71–91.

[Moran et al., 2023] Moran, M. B., Vasconcellos, E. C., Cuno, J. J., Biondi, M., Riveaux, J. M., Correia, M. D., Clua, E. W. G., and Conci, A. (2023). Heuristic-based approaches for fracture detection in borehole images. *International Journal of Innovative Computing and Applications*, 14(1-2), 78–90.

[Oktay et al., 2018] Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N. Y., Kainz, B., Glocker, B., and Rueckert, D. (2018). Attention U-Net: Learning where to look for the pancreas. arXiv:1804.03999.

[Perritaz, 2024] Perritaz, P. (2024). Thesis proposal: Automated structure picking of optical and acoustic televiewer borehole images using machine learning. D-ERDW, ETH Zürich.

[Plenkers et al., 2023] Plenkers, K., Reinicke, A., Obermann, A., Gholizadeh Doonechaly, N., Krietsch, H., Fechner, T., Hertrich, M., Kontar, K., Maurer, H., Philipp, J., et al. (2023). Multi-disciplinary monitoring networks for mesoscale underground experiments: Advances in the Bedretto Reservoir Project. *Sensors*, 23(6), 3315.

[Prince, 2023] Prince, S. J. (2023). *Understanding Deep Learning*. MIT Press.

[Ronneberger et al., 2015] Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *MICCAI 2015*, Lecture Notes in Computer Science vol. 9351, pages 234–241. Springer. arXiv:1505.04597.

[Stigsson & Munier, 2013] Stigsson, M. and Munier, R. (2013). Orientation uncertainty goes bananas: An algorithm to visualise the uncertainty sample space on stereonets for oriented objects measured in boreholes. *Computers & Geosciences*, 56, 56–61.

[Stigsson, 2024] Stigsson, M. (2024). Personal communication with Martin Stigsson.

[Sudre et al., 2017] Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., and Jorge Cardoso, M. (2017). Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations. In *MICCAI Workshop on Deep Learning in Medical Image Analysis (DLMIA 2017)*, Lecture Notes in Computer Science, pages 240–248. Springer. arXiv:1707.03237.

[Thapa et al., 1997] Thapa, B. B., Hughett, P., and Karasaki, K. (1997). Semi-automatic analysis of rock fracture orientations from borehole wall images. *Geophysics*, 62(1), 129–137.
