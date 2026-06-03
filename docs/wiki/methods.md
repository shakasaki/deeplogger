# Methods

*Paper section: §4 Methods*

---

## 1. Model Architecture

All models follow the 2D U-Net template [@ronneberger2015]: an encoder (contracting path)
progressively downsamples the input via pooling layers, a bottleneck processes the most
compressed representation, and a decoder (expansive path) restores spatial resolution via
transposed convolutions. Skip connections concatenate encoder feature maps directly to
the corresponding decoder level, preserving fine spatial detail.

See `docs/architecture_diagram.png` for block diagrams of both v2 architectures.

### 1.1 Shared Building Block

All encoder/decoder stages use the same **double-convolution block** (after [@perritaz2024]):

```
Conv2d(k=3, p=1) → BatchNorm2d → ReLU
Conv2d(k=3, p=1) → BatchNorm2d → ReLU
```

Bias is disabled in Conv2d layers (BatchNorm subsumes it). `init_features = 32` throughout
(channel counts: 32 → 64 → 128 → 256, bottleneck 512).

Implementation: `_conv_block()` in `deeplogger/model_architectures_v2.py`.

### 1.2 UNetATV v1 — Perritaz Baseline

**File:** `deeplogger/model_architectures_ATV.py`

Faithful reproduction of the architecture described in [@perritaz2024, §4.4.1]:

| Stage | Spatial op | Channels |
|---|---|---|
| Encoder 1–3 | MaxPool2d(k=2, s=2) | 32, 64, 128 |
| Encoder 4 | MaxPool2d(**k=3, s=1, p=1**) — size-preserving | 256 |
| Bottleneck | — | 512 |
| Decoder 4–1 | ConvTranspose2d(k=2, s=2) | 256, 128, 64, 32 |
| Head | Conv2d(k=1) + Sigmoid | 1 |

The size-preserving pool4 (stride=1) is an architectural quirk: it does not downsample,
giving the bottleneck a wider receptive field at the cost of a skip-connection alignment
mismatch (skip3/4 require bilinear interpolation; skip1/2 require H,W divisible by 4).

Parameters: ~7.76M.

### 1.3 UNetATV v2 — Perritaz + Normalisation

**File:** `deeplogger/model_architectures_v2.py` · **Class:** `UNetATV`

Identical to v1, with two modifications:

1. **Per-sample min-max normalisation** applied at the start of `forward()` (see [dataset.md §5.2](dataset.md#52-amplitude-normalisation)):
   $$\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x) + \varepsilon}$$
   Each sample in the batch is normalised independently, removing sensitivity to absolute
   ATV amplitude scale.

2. **`F.interpolate` on all four skip connections**, so any input (H, W) is accepted
   without divisibility constraints.

Parameters: ~7.76M.

### 1.4 AttentionUNetATV — Attention + Dropout

**File:** `deeplogger/model_architectures_v2.py` · **Class:** `AttentionUNetATV`

Extended architecture with three improvements over v2, motivated by the class imbalance
and limited dataset size:

#### 1.4.1 Attention Gates [@oktay2018]

At each decoder level, before concatenating the encoder skip feature $\mathbf{x}$ with the
upsampled decoder signal $\mathbf{g}$, a soft attention gate computes:

$$\alpha = \sigma\!\left(\psi\!\left(\text{ReLU}\!\left(W_g\,\mathbf{g} + W_x\,\mathbf{x}\right)\right)\right) \in (0,1)$$

$$\mathbf{x}' = \mathbf{x} \odot \alpha$$

where $W_g$, $W_x$ are 1×1 convolutions projecting to an intermediate dimension
$F_\text{int} = F_x / 2$, $\psi$ is a 1×1 conv to a scalar, and $\sigma$ is sigmoid.
The gate suppresses background-dominated skip features before they enter the decoder,
directly addressing the <1 % foreground imbalance. Implementation: `AttentionGate` class.

#### 1.4.2 Spatial Dropout (Dropout2d)

`nn.Dropout2d` randomly zeros entire feature map channels during training. Applied to:

- Encoder 3: p = 0.1
- Encoder 4: p = 0.1
- Bottleneck: p = 0.2

Spatial dropout prevents co-adaptation between feature detectors and reduces overfitting
on the ~1700-sample dataset. It is more effective than element-wise dropout for
convolutional networks because it drops correlated spatial information [@srivastava2014].

#### 1.4.3 Real Pool4 (stride = 2)

The size-preserving pool4 (stride=1) from v1/v2 is replaced with a genuine
`MaxPool2d(k=2, s=2)`, giving a clean 4-level encoder (bottleneck at H/16, W/16).
This removes the spatial alignment quirk.

Parameters: ~7.89M (+130K for attention gates over v2).

### 1.5 Architecture Comparison

| Feature | v1 | v2 | Attention |
|---|---|---|---|
| Per-sample normalisation | ✗ | ✓ | ✓ |
| Attention gates on skips | ✗ | ✗ | ✓ |
| Spatial Dropout2d | ✗ | ✗ | ✓ |
| Pool4 stride | 1 (no-op) | 1 (no-op) | 2 (real) |
| Bottleneck depth | H/8 | H/8 | H/16 |
| Any (H,W) input | ✗ | ✓ | ✓ |
| Parameters | 7.76M | 7.76M | 7.89M |

---

## 2. Loss Functions

### 2.1 Binary Cross-Entropy (BCE)

$$\mathcal{L}_\text{BCE}(y, \hat{y}) = -\left[y \log \hat{y} + (1-y)\log(1-\hat{y})\right]$$

Penalises per-pixel errors equally. Suffers under extreme imbalance: predicting all-background
achieves >99 % accuracy and low BCE, but zero sensitivity [@janthakal2021].

### 2.2 Dice Loss

$$\mathcal{L}_\text{Dice}(y, \hat{y}) = 1 - \frac{2\sum y\hat{y} + \varepsilon}{\sum y + \sum \hat{y} + \varepsilon}$$

where $\varepsilon = 1$ (Laplace smoothing) and the sum is over all pixels in the batch.
Directly optimises foreground overlap, robust to imbalance [@sudre2017].
$\mathcal{L}_\text{Dice} = 0$ when $y = \hat{y}$; $\approx 1$ when there is no overlap.

### 2.3 BCE+Dice Combined Loss

$$\mathcal{L} = w_\text{BCE} \cdot \mathcal{L}_\text{BCE} + w_\text{Dice} \cdot \mathcal{L}_\text{Dice}$$

Default weights: $w_\text{BCE} = w_\text{Dice} = 0.5$ [@alexakis2020; @sudre2017].
BCE stabilises early training via dense pixel gradients; Dice forces the model to improve
foreground recall as training progresses. This combination was found most effective for
OTV data in [@perritaz2024]; recommended for ATV retraining.

Implementation: `BCEDiceLoss` in `deeplogger/loss_functions.py`. Shape-agnostic: accepts
`(B, H, W)` or `(B, 1, H, W)`.

---

## 3. Training Strategy

### 3.1 Configuration

Full TrainingConfig in `deeplogger/config.py`. Recommended settings for ATV retraining:

| Parameter | Recommended | Thesis baseline |
|---|---|---|
| Architecture | `AttentionUNetATV` | `UNetATV v1` |
| Loss | `BCE_DICE` | `BCE` |
| Optimizer | `Adam` | `Adam` |
| Learning rate | 5×10⁻⁴ | 0.1 |
| Epochs | 200 | up to 600 (best at 75) |
| Batch size | 20 | 20 |
| LR scheduler | StepLR(γ=0.75, step=1) | StepLR(γ=0.75, step=1) |

For BCE+Dice, [@alexakis2020] recommend Adam with lr = 5×10⁻⁴. For BCE alone, the thesis
used Adam with lr = 0.1 (aggressive but effective at epoch 75).

### 3.2 Optimizers

**Adam** [@kingma2014]: adaptive learning rate per parameter; recommended for BCE+Dice.
Converged to best ATV performance at epoch 75, faster than SGDM [@perritaz2024, §5.1].

**SGD with momentum** (η = 0.9): used in thesis for OTV with Dice and BCE.
Momentum smooths gradient oscillations ("edge of stability") [@fu2023].

### 3.3 Augmentation

Random horizontal flip (p = 0.5) applied jointly to image and mask. Horizontal flip
corresponds to a 180° rotation of the borehole azimuth, which is geometrically valid.
Vertical flip is not used (reverses depth ordering).

### 3.4 Validation and Checkpointing

Validation runs every 15 epochs (`validate_every=15`). Best model checkpoint saved
whenever validation loss improves. Final config + loss curves serialised to `.p` pickle.

---

## 4. Evaluation Metrics

Test set evaluation at thresholds θ = 0.5 and θ = 0.75:

| Metric | Formula | Notes |
|---|---|---|
| Accuracy | $(TP + TN) / N$ | Misleading under imbalance |
| Sensitivity (Recall) | $TP / (TP + FN)$ | Primary: fracture detection rate |
| Specificity | $TN / (TN + FP)$ | Trivially high (background dominates) |
| Precision | $TP / (TP + FP)$ | Expected low under imbalance [@li2019overfitting] |
| F1 / Dice | $2TP / (2TP + FP + FN)$ | Primary: foreground overlap |
| Mean per-image Dice | mean of per-image F1 | Less sensitive to batch size |

Sensitivity and F1 are the primary metrics. Precision is expected to remain low (<10 %)
due to imbalance — the thesis reports 2 % ± 3–9 % [@perritaz2024, Table 4], consistent
with the theoretical analysis in [@li2019overfitting].

---

## 5. Inference Pipeline

1. Load ATV log (LAS or zarr cache)
2. Apply SVD filter with **k = 3** (mandatory for models trained on Perritaz data)
3. Extract depth window (≤ 20k rows recommended for the GUI)
4. Pad H and W to satisfy alignment constraints (OTV: `%16==8`; ATV v2/attention: any size)
5. Run `model.forward(image)` → probability map in [0, 1]
6. Apply threshold (default 0.5) → binary fracture mask
7. Overlay on original image in napari; user edits if needed

---

## References (this page)

`[@ronneberger2015]`, `[@oktay2018]`, `[@sudre2017]`, `[@alexakis2020]`,
`[@perritaz2024]`, `[@li2019overfitting]`, `[@janthakal2021]`, `[@fu2023]`,
`[@kingma2014]`, `[@srivastava2014]`

Full BibTeX: [`references.bib`](references.bib)
