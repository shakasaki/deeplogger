# Background

*Paper sections: §1 Introduction, §2 Related Work*

---

## 1. Motivation

Borehole televiewer logging is a standard technique for characterising geological structures
(fractures, bedding planes, fault zones) in deep boreholes. A televiewer rotates continuously
during extraction, producing a 360° panoramic image of the borehole wall. When the image is
unrolled azimuthally, planar structures appear as sinusoidal traces — a direct consequence of
the intersection geometry between a plane and a cylinder [@gaillot2007].

Manual picking of these sinusoids is the current standard workflow. It is performed by domain
experts in commercial software (WellCAD™) and is:

- **Time-consuming:** a single 100 m log section may contain hundreds of fractures
- **Subjective:** different interpreters pick different subsets
- **Non-scalable:** the BedrettoLab alone has 16 boreholes with thousands of metres of logs

Automated detection would accelerate fracture characterisation, reduce interpreter bias, and
enable monitoring workflows that require near-real-time structure tracking.

---

## 2. Problem Statement

The task is **binary semantic segmentation** of borehole televiewer images: classify each
pixel as fracture (foreground, label 1) or background (label 0).

Key challenges:

1. **Extreme class imbalance.** Fractures occupy <1 % of pixels. Standard cross-entropy loss
   trained naively on such data converges to predict all-background [@li2019overfitting].

2. **Two data modalities with different characteristics:**
   - **ATV (Acoustic Televiewer):** single-channel float amplitude; strong coherent vertical
     stripes from tool rotation artifacts; amplitude scale varies between boreholes and runs.
   - **OTV (Optical Televiewer):** three-channel RGB; no stripe artifacts; lower structural
     contrast in intact rock.

3. **Limited training data.** The Perritaz (2024) dataset contains 1709 manually labelled ATV
   snippets [@perritaz2024]. This is small relative to standard deep learning benchmarks
   (e.g. PASCAL VOC at 11 000+ images [@flickr2005]).

4. **Variable snippet geometry.** Snippets are 360 px wide (one full azimuth revolution) and
   nominally 360 px tall, but represent 0.7–1.8 m of depth depending on the vertical resolution
   of the logging instrument [@perritaz2024].

---

## 3. Related Work

### 3.1 Classical Methods

Early automated approaches used the Hough transform to detect sinusoidal curves in borehole
images [@glossop1999; @thapa1997]. [@assous2014] automated planar feature detection using
template matching. [@moran2020; @moran2023] applied iterated local search heuristics.
These methods are fragile to noise and require expert-tuned parameters for each logging run.

### 3.2 Deep Learning on Borehole Images

[@dias2020] applied Fast R-CNN to detect fracture bounding boxes in acoustic borehole logs,
using a dataset of ~10 000 images from oil-well logging. This is ~6× larger than the Bedretto
dataset. [@han2023] combined Faster R-CNN with the Hough transform to extract sine fits from
detected structures, achieving 91.5 % accuracy on an unlabelled test set. Neither work provides
pixel-level segmentation masks.

### 3.3 U-Net for Semantic Segmentation

The U-Net architecture [@ronneberger2015] was developed for biomedical image segmentation,
precisely the regime of limited labelled data and pixel-level output required here. It combines
a contracting encoder (context) with an expansive decoder (localisation) connected by skip
connections that preserve spatial detail. [@alexakis2020] evaluated U-Net and U-Net++ for
high-resolution change detection, demonstrating that the architecture transfers well to
geospatial data. [@perritaz2024] applied 2D U-Net directly to ATV and OTV borehole logs —
the direct precursor to this work.

### 3.4 Attention Mechanisms

[@oktay2018] introduced the Attention U-Net, adding soft spatial attention gates on skip
connections. A gating signal from the decoder computes a per-location weight map α ∈ (0, 1)
that suppresses irrelevant background features before they enter the decoder. This is
particularly motivated for imbalanced data: the gate learns to focus on foreground-likely
regions, improving sensitivity without sacrificing specificity. [@li2019overfitting] showed
that class imbalance causes neural networks to overfit background statistics and that
foreground-sensitive losses and architectural attention both help.

### 3.5 Loss Functions for Imbalanced Segmentation

[@sudre2017] proposed the Generalised Dice Loss as a training objective directly optimising
overlap between predicted and ground-truth regions, which is robust to class imbalance by
normalising by foreground area rather than pixel count. [@alexakis2020] combined BCE and Dice
losses for multi-class change detection. [@perritaz2024] tested BCE, Dice, and BCE+Dice on
the Bedretto dataset; BCE+Dice with Adam was most effective for the OTV case but unstable
for ATV (attributed to data quality; see [dataset.md](dataset.md)).

### 3.6 Gaps Addressed by This Work

| Gap | This work |
|---|---|
| Pixel-level ATV segmentation at BedrettoLab scale | ✓ |
| Architecture comparison (baseline vs attention + dropout) | ✓ (UNetATV v2 vs AttentionUNetATV) |
| Per-sample amplitude normalisation for ATV | ✓ (`_norm_input` inside `forward()`) |
| Human-in-the-loop correction + fine-tuning | ✓ (napari correction loop + `finetune()`) |
| Open-source, reproducible training pipeline | ✓ (`scripts/run_training.py`) |

---

## References (this page)

Full BibTeX in [`references.bib`](references.bib). Short keys used inline:

- `[@gaillot2007]` → [@gaillot2007]  
- `[@perritaz2024]` → [@perritaz2024]  
- `[@ronneberger2015]` → [@ronneberger2015]  
- `[@oktay2018]` → [@oktay2018]  
- `[@sudre2017]` → [@sudre2017]  
- `[@alexakis2020]` → [@alexakis2020]  
- `[@han2023]` → [@han2023]  
- `[@dias2020]` → [@dias2020]  
- `[@li2019overfitting]` → [@li2019overfitting]  
- `[@assous2014]` → [@assous2014]  
- `[@plenkers2023]` → [@plenkers2023]  
