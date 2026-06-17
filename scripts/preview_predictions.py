#!/usr/bin/env python3
"""Export visual prediction previews from a trained DeepLogger model.

This script loads a saved training config (.p), reuses the held-out test IDs from
that run, runs inference with the corresponding checkpoint, and writes per-sample
figures showing input, ground truth, prediction heatmap, and overlay.
"""

import argparse
import csv
import math
import os
import pickle
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplogger.config import TrainingConfig
from deeplogger.dataloader import BoreholeDataset
from deeplogger.train import _binarize_mask, _build_model, _ensure_image_nchw


ATV_PREVIEW_CMAP = "afmhot"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-path", required=True, help="Path to *_config.p from training")
    p.add_argument("--model-path", default=None, help="Path to *_best.pt checkpoint (default: inferred from config path)")
    p.add_argument("--output-dir", default=None, help="Directory for output PNGs (default: sibling folder prediction_previews)")
    p.add_argument("--n-samples", type=int, default=12, help="Number of test samples to visualize")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary overlay and Dice")
    p.add_argument("--seed", type=int, default=100, help="Sampling seed")
    p.add_argument("--contact-cols", type=int, default=3, help="Number of columns in the contact sheet")
    return p.parse_args()


def infer_model_path(config_path: str) -> str:
    base = os.path.basename(config_path)
    if base.endswith("_config.p"):
        return os.path.join(os.path.dirname(config_path), base.replace("_config.p", "_best.pt"))
    return os.path.join(os.path.dirname(config_path), "model_best.pt")


def _resolve_existing_path(path: str, config_path: str) -> str:
    if os.path.exists(path):
        return path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    candidates = [
        os.path.normpath(os.path.join(os.getcwd(), path)),
        os.path.normpath(os.path.join(script_dir, path)),
        os.path.normpath(os.path.join(repo_root, path)),
        os.path.normpath(os.path.join(cfg_dir, path)),
        os.path.normpath(os.path.join(cfg_dir, "..", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _to_display_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    p_low, p_high = np.percentile(img, [1, 99])
    if p_high <= p_low:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - p_low) / (p_high - p_low), 0.0, 1.0)


def _dice(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred_b = (pred >= threshold).astype(np.float32)
    gt_b = (gt > 0).astype(np.float32)
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * (pred_b * gt_b).sum()) / denom)


def _write_summary_csv(rows: list[dict], output_path: str) -> None:
    fieldnames = [
        "index",
        "source_pt",
        "preview_png",
        "dice",
        "ground_truth_pixels",
        "predicted_pixels",
        "threshold",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_contact_sheet(rows: list[dict], output_path: str, n_cols: int) -> None:
    if not rows:
        return

    n_cols = max(1, n_cols)
    n_rows = math.ceil(len(rows) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, rows):
        img = plt.imread(row["preview_png"])
        ax.imshow(img)
        ax.set_title(f"{os.path.basename(row['source_pt'])}\nDice={row['dice']:.3f}")
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle("DeepLogger Prediction Preview Contact Sheet", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    model_path = args.model_path or infer_model_path(args.config_path)

    with open(args.config_path, "rb") as f:
        run_data = pickle.load(f)

    if "test_ids" not in run_data or not run_data["test_ids"]:
        raise ValueError("No test_ids found in config file. Cannot build evaluation previews.")

    config = TrainingConfig.from_dict(run_data)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.config_path), "prediction_previews")
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    test_ids = [_resolve_existing_path(p, args.config_path) for p in run_data["test_ids"]]
    missing = [p for p in test_ids if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Could not resolve {len(missing)} test file paths. Example: {missing[0]}")
    n = min(args.n_samples, len(test_ids))
    selected_ids = rng.sample(test_ids, n)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    dataset = BoreholeDataset(selected_ids, device=device)

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Config: {args.config_path}")
    print(f"Saving {n} preview(s) to: {output_dir}")

    summary_rows = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img_t, mask_t = dataset[i]

            img_b = _ensure_image_nchw(img_t.unsqueeze(0))
            pred = model(img_b).squeeze().detach().cpu().numpy().astype(np.float32)

            mask = _binarize_mask(mask_t).squeeze().detach().cpu().numpy().astype(np.float32)
            img = img_t.detach().cpu().numpy()
            img_disp = _to_display_image(img)
            dice = _dice(pred, mask, args.threshold)

            pred_bin = (pred >= args.threshold).astype(np.float32)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
            axes[0].imshow(img_disp, cmap=ATV_PREVIEW_CMAP, aspect="auto")
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(mask, cmap="Reds", vmin=0, vmax=1, aspect="auto")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            im = axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1, aspect="auto")
            axes[2].set_title("Prediction Prob")
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046)

            axes[3].imshow(img_disp, cmap=ATV_PREVIEW_CMAP, aspect="auto")
            axes[3].imshow(pred_bin, cmap="Reds", alpha=0.4, vmin=0, vmax=1, aspect="auto")
            axes[3].set_title(f"Overlay (Dice={dice:.3f})")
            axes[3].axis("off")

            src = os.path.basename(selected_ids[i]).replace(".pt", "")
            out_path = os.path.join(output_dir, f"{i:02d}_{src}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            summary_rows.append({
                "index": i,
                "source_pt": selected_ids[i],
                "preview_png": out_path,
                "dice": dice,
                "ground_truth_pixels": int((mask > 0).sum()),
                "predicted_pixels": int(pred_bin.sum()),
                "threshold": args.threshold,
            })
            print(f"[{i + 1}/{n}] {out_path}  dice={dice:.3f}")

    csv_path = os.path.join(output_dir, "prediction_summary.csv")
    _write_summary_csv(summary_rows, csv_path)
    print(f"Summary CSV: {csv_path}")

    contact_path = os.path.join(output_dir, "prediction_contact_sheet.png")
    _write_contact_sheet(summary_rows, contact_path, args.contact_cols)
    print(f"Contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
