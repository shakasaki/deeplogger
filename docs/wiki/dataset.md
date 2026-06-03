# Dataset

*Paper section: §3 Data*

---

## 1. Study Site — BedrettoLab

The Bedretto Underground Laboratory (BedrettoLab) is located in the Rotondo Granite,
~1 km below the surface in the Bedretto tunnel, Ticino, Switzerland
[@ma2021; @plenkers2023]. It hosts ongoing experiments in geothermal energy, fault
mechanics, and induced seismicity. Its geological setting — fractured crystalline rock —
is representative of enhanced geothermal system (EGS) targets.

Sixteen boreholes were drilled from the tunnel at depths up to ~300 m [@castilla2022].
ATV and OTV logging was performed in most boreholes using ALT ABI40 (acoustic) and
OBI40 (optical) instruments. The target fault for fluid-induced activation experiments
is described in [@achtziger2024]. Stress field characterisation from hydraulic fracture
traces is reported in [@broker2024].

---

## 2. Instruments

### 2.1 Acoustic Televiewer (ATV)

The ATV emits focused ultrasonic pulses along the borehole wall and records two
quantities per azimuth bin [@gaillot2007]:

- **Amplitude** [arbitrary units]: reflection coefficient of the wall surface; open fractures
  appear as dark (low-amplitude) bands because fluid-filled apertures scatter/absorb acoustic
  energy.
- **Travel time** [µs]: one-way propagation time to the wall; used to compute borehole radius.

Only the amplitude image is used for fracture segmentation in this work. The ATV is preferable
to OTV in turbid borehole fluids (common in Bedretto) where optical imaging fails [@gaillot2007].

**Output format:** single-channel float32, azimuthal resolution 360 px, vertical resolution
varies by logging speed (typically 0.5–1 mm/px in the Bedretto data).

### 2.2 Optical Televiewer (OTV)

The OTV captures a 3-channel RGB panoramic photograph of the borehole wall under ring-flash
illumination. Fractures appear as darker sinusoidal bands in the unrolled image. The OTV
requires clear (non-turbid) borehole fluid and provides colour/texture information unavailable
to the ATV.

**Output format:** 3-channel uint8 RGB, azimuthal resolution 360 px (reshaped to 360 for
training if different), vertical resolution varies.

---

## 3. Data Format

All data is stored in LAS v3.0 format. The `deeplogger` LAS reader
(`deeplogger/las_reader.py`) parses STRT/STOP/STEP/NULL/delimiter from the header and
returns:

- ATV: `(N, 360)` float32, NULL values replaced with NaN
- OTV: `(N, 360, 3)` uint8 RGB

Training snippets are 360 × 360 px subimages stored as PyTorch `.pt` files containing
`[image_tensor, mask_tensor]`. Image shape: ATV `(1, 360, 360)`, OTV `(3, 360, 360)`.
Mask shape: `(1, 360, 360)`, integer labels.

### Label Encoding

| Value | Meaning |
|---|---|
| 0 | Background (borehole wall, no structure) |
| 1 | Background (legacy encoding — mapped to 0 during training) |
| 2 | Fracture / geological structure (foreground) |
| 3 | Fracture secondary class (foreground; one labelling error noted in thesis) |

During training `_binarize_mask()` maps: 1→0, 2→1, 3→1.

---

## 4. Training Dataset

### 4.1 Composition

| Modality | Snippets | Source boreholes | Labels |
|---|---|---|---|
| ATV | 1709 | Bedretto boreholes (BFE, SB series) | Manual (WellCAD™ + rasterization) |
| OTV | — | Same | Manual + automatic (sine rasterization) |

Data path: `~/DATA/Bedretto varia/Data_Msc_Thesis_Perritaz/data/Training_data_manually_Drawn_labels/atv_data_label/`

### 4.2 Label Generation

Two labelling approaches were used [@perritaz2024, §4.3.3]:

**Automatic labels:** fracture picks (depth, dip, azimuth, aperture) from expert interpretation
in WellCAD™ were rasterized into binary masks using a sinusoidal model
(`deeplogger/labels.py:get_label`). This produces geometrically perfect but potentially
inaccurate masks where picks deviate from the actual image structure.

**Manual labels:** experts painted directly on the ATV amplitude image, pixel by pixel. These
are more accurate but more time-consuming. The Perritaz (2024) thesis found that manual labels
were necessary for successful ATV model training — automatic labels alone did not yield
convergent training curves.

### 4.3 Class Imbalance

Fractures are typically <1 % of pixels in a 360 × 360 snippet. This extreme imbalance is
the primary challenge for model training:

- Specificity (background recall) is trivially high (>99 %) for any model
- Sensitivity (fracture recall) and precision are the meaningful metrics
- F1 score and Dice coefficient are the primary evaluation criteria

See [@li2019overfitting] for a theoretical analysis of class imbalance effects on
segmentation networks.

---

## 5. Preprocessing Pipeline

### 5.1 SVD Destriping (ATV only)

ATV amplitude images contain coherent vertical stripes from tool rotation artifacts and
electronic noise. Singular Value Decomposition (SVD) filtering removes these:

$$\mathbf{X}_{\text{filt}} = \mathbf{X} - \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

where $\sigma_i$, $\mathbf{u}_i$, $\mathbf{v}_i$ are the $i$-th singular value and vectors.

**k = 3 was used for all training data** [@perritaz2024, §4.3.2, p. 14 of PDF]:

> "The acoustic data was pre-processed by applying a singular value decomposition (SVD)
> filter, removing the largest three singular values. Prevalent noise traces could be
> withdrawn, resulting in clearer data images."

**Consequence for inference:** new ATV data passed to any model trained on this dataset
must also be SVD-filtered with k = 3 before inference. Failing to do so creates a
distribution mismatch between training and test data. The GUI viewer (`LogViewer`) exposes
a spinbox for k; set it to 3 and save the processed zarr before running inference.

Implementation: `deeplogger/image_processing.py:remove_svd_components(image, n_components=3)`.

### 5.2 Amplitude Normalisation

Raw ATV amplitude values vary substantially between boreholes and logging campaigns due to
instrument gain settings and borehole fluid properties. V2 models (`UNetATV`, `AttentionUNetATV`)
apply **per-sample min-max normalisation** inside `forward()`:

$$\hat{x} = \frac{x - \min(x)}{\max(x) - \min(x) + \varepsilon}, \quad \varepsilon = 10^{-8}$$

This is applied independently per sample in the batch, making the model invariant to
absolute amplitude scale. No external normalisation step is required — it is automatic.

The V1 model (`UNetATV_v1`, Perritaz baseline) does not normalise internally.

### 5.3 Snippet Creation

Snippets are created by sliding a 360 × 360 px window periodically along the borehole log.
Sections with excessive noise (typically near the top of a log) are excluded manually.
The vertical extent of a 360-px snippet ranges from 0.7 to 1.8 m depending on the
instrument's vertical pixel resolution [@perritaz2024, §4.3.2].

---

## 6. Dataset Split

**Current (random):** 10 % test (`test_fraction=0.1`), 20 % of remainder for validation
(`val_fraction=0.2`), 70 % for training.

**Known issue:** random splitting of snippets from the same borehole creates data leakage.
Spatially adjacent snippets (same depth range, same borehole) appear in both train and test
sets, inflating test performance. The correct approach is **borehole-stratified splitting**:
hold out complete boreholes (SB 2.3 and SB 3.1 were used as the test boreholes in
[@perritaz2024]) and train on the remainder.

This is a pending TODO in `train.py`.

---

## References (this page)

`[@ma2021]`, `[@plenkers2023]`, `[@castilla2022]`, `[@achtziger2024]`, `[@broker2024]`,
`[@gaillot2007]`, `[@perritaz2024]`, `[@li2019overfitting]`

Full BibTeX: [`references.bib`](references.bib)
