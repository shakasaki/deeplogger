#!/usr/bin/env python3
"""Compare multiple trained models on shared held-out test samples.

Exports one figure per sample with input/ground-truth plus one overlay column per
model, and writes CSV summaries (per-sample + per-model averages).
"""

import argparse
import csv
import os
import pickle
import random
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplogger.config import TrainingConfig
from deeplogger.train import _binarize_mask, _build_model, _ensure_image_nchw

ATV_PREVIEW_CMAP = "afmhot"


@dataclass
class ModelSpec:
    label: str
    config_path: str
    model_path: str


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: label::config_path::model_path (repeat for multiple models)",
    )
    p.add_argument("--output-dir", required=True, help="Directory for output figures and CSV summaries")
    p.add_argument("--n-samples", type=int, default=12, help="Number of shared test samples to visualize")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary overlay and Dice")
    p.add_argument("--seed", type=int, default=100, help="Sampling seed")
    return p.parse_args()


def _resolve_existing_path(path: str, anchor_path: str) -> str:
    if os.path.exists(path):
        return path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    anchor_dir = os.path.dirname(os.path.abspath(anchor_path))
    candidates = [
        os.path.normpath(os.path.join(os.getcwd(), path)),
        os.path.normpath(os.path.join(script_dir, path)),
        os.path.normpath(os.path.join(repo_root, path)),
        os.path.normpath(os.path.join(anchor_dir, path)),
        os.path.normpath(os.path.join(anchor_dir, "..", path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid --model spec: {raw}")
    label, config_path, model_path = parts
    return ModelSpec(label=label.strip(), config_path=config_path.strip(), model_path=model_path.strip())


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


def _load_run_data(config_path: str) -> tuple[TrainingConfig, dict]:
    with open(config_path, "rb") as f:
        run_data = pickle.load(f)
    config = TrainingConfig.from_dict(run_data)
    return config, run_data


def _shared_test_ids(specs: list[ModelSpec]) -> list[str]:
    shared = None
    for spec in specs:
        _, run_data = _load_run_data(spec.config_path)
        if "test_ids" not in run_data or not run_data["test_ids"]:
            raise ValueError(f"No test_ids found in {spec.config_path}")
        resolved = {_resolve_existing_path(p, spec.config_path) for p in run_data["test_ids"]}
        missing = [p for p in resolved if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Could not resolve test path for {spec.label}. Example: {missing[0]}")
        shared = resolved if shared is None else (shared & resolved)

    if not shared:
        raise ValueError("No shared test IDs across selected models.")
    return sorted(shared)


def _load_models(specs: list[ModelSpec], device: torch.device):
    loaded = []
    for spec in specs:
        config, _ = _load_run_data(spec.config_path)
        model = _build_model(config, device)
        state = torch.load(spec.model_path, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        loaded.append((spec.label, model))
    return loaded


def main():
    args = parse_args()
    specs = [_parse_model_spec(raw) for raw in args.model]

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shared_ids = _shared_test_ids(specs)
    n = min(args.n_samples, len(shared_ids))
    rng = random.Random(args.seed)
    selected_ids = rng.sample(shared_ids, n)

    print(f"Device: {device}")
    print(f"Models: {[s.label for s in specs]}")
    print(f"Shared test pool: {len(shared_ids)}")
    print(f"Selected samples: {n}")
    print(f"Output dir: {args.output_dir}")

    loaded_models = _load_models(specs, device)

    rows = []

    with torch.no_grad():
        for i, sample_path in enumerate(selected_ids):
            sample = torch.load(sample_path, map_location=device, weights_only=False)
            img_t = sample[0].to(device).float() if isinstance(sample[0], torch.Tensor) else torch.from_numpy(np.asarray(sample[0], dtype=np.float32)).to(device)
            mask_t = sample[1].to(device).float() if isinstance(sample[1], torch.Tensor) else torch.from_numpy(np.asarray(sample[1], dtype=np.float32)).to(device)

            img_b = _ensure_image_nchw(img_t.unsqueeze(0))
            mask = _binarize_mask(mask_t).squeeze().detach().cpu().numpy().astype(np.float32)
            img_disp = _to_display_image(img_t.detach().cpu().numpy())

            n_cols = 2 + len(loaded_models)
            fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4), constrained_layout=True)

            axes[0].imshow(img_disp, cmap=ATV_PREVIEW_CMAP, aspect="auto")
            axes[0].set_title("Input")
            axes[0].axis("off")

            axes[1].imshow(mask, cmap="Reds", vmin=0, vmax=1, aspect="auto")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            for j, (label, model) in enumerate(loaded_models, start=2):
                pred = model(img_b).squeeze().detach().cpu().numpy().astype(np.float32)
                pred_bin = (pred >= args.threshold).astype(np.float32)
                dice = _dice(pred, mask, args.threshold)

                axes[j].imshow(img_disp, cmap=ATV_PREVIEW_CMAP, aspect="auto")
                axes[j].imshow(pred_bin, cmap="Reds", alpha=0.4, vmin=0, vmax=1, aspect="auto")
                axes[j].set_title(f"{label}\nDice={dice:.3f}")
                axes[j].axis("off")

                rows.append({
                    "sample_index": i,
                    "source_pt": sample_path,
                    "model_label": label,
                    "dice": dice,
                    "threshold": args.threshold,
                    "ground_truth_pixels": int((mask > 0).sum()),
                    "predicted_pixels": int(pred_bin.sum()),
                })

            src = os.path.basename(sample_path).replace(".pt", "")
            out_path = os.path.join(args.output_dir, f"{i:02d}_{src}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[{i + 1}/{n}] {out_path}")

    per_sample_csv = os.path.join(args.output_dir, "comparison_per_sample.csv")
    with open(per_sample_csv, "w", newline="") as f:
        fieldnames = [
            "sample_index",
            "source_pt",
            "model_label",
            "dice",
            "threshold",
            "ground_truth_pixels",
            "predicted_pixels",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_model = {}
    for row in rows:
        by_model.setdefault(row["model_label"], []).append(float(row["dice"]))

    summary_csv = os.path.join(args.output_dir, "comparison_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        fieldnames = ["model_label", "n_samples", "mean_dice", "min_dice", "max_dice"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, vals in sorted(by_model.items()):
            writer.writerow({
                "model_label": label,
                "n_samples": len(vals),
                "mean_dice": float(np.mean(vals)),
                "min_dice": float(np.min(vals)),
                "max_dice": float(np.max(vals)),
            })

    print(f"Per-sample CSV: {per_sample_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
