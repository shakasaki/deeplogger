#!/usr/bin/env python3
"""Train one or more DeepLogger models and write a markdown evaluation report.

Usage
-----
# Recommended ATV run (BCE+Dice, Adam):
python scripts/run_training.py \\
    --data-dir ~/DATA/.../atv_data_label/ \\
    --model-dir models/ \\
    --models unet_atv_v2 attention_unet_atv \\
    --loss bce_dice --lr 5e-4 --epochs 200

# Quick smoke-test on a small dir:
python scripts/run_training.py \\
    --data-dir /tmp/test_bundles/ \\
    --models unet_atv_v2 \\
    --epochs 5

The script trains each requested model in sequence on the same data split, then
evaluates the best checkpoint on the held-out test set and writes a markdown
report to <model-dir>/<run-name>_report.md.
"""

import argparse
import datetime
import os
import subprocess
import sys

import numpy as np
import torch
import torch.utils.data as data

# Ensure repo root is importable when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deeplogger.config import (
    DataType,
    LossType,
    ModelType,
    OptimizerType,
    TrainingConfig,
)
from deeplogger.dataloader import BoreholeDataset
from deeplogger.train import _binarize_mask, _build_model, train


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_MODEL_CHOICES = {
    "unet_atv_v1":       ModelType.UNET_ATV_V1,
    "unet_atv_v2":       ModelType.UNET_ATV_V2,
    "attention_unet_atv": ModelType.ATTENTION_UNET_ATV,
    "unet_otv":          ModelType.UNET_OTV,
}
_LOSS_CHOICES = {
    "bce":       LossType.BCE,
    "dice":      LossType.DICE,
    "bce_dice":  LossType.BCE_DICE,
}
_OPT_CHOICES = {
    "adam": OptimizerType.ADAM,
    "sgd":  OptimizerType.SGD,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",  required=True, help="Directory with .pt training bundles")
    p.add_argument("--model-dir", default="models", help="Where to save weights + report (default: models/)")
    p.add_argument("--data-type", choices=["atv", "otv"], default="atv", help="ATV (1-ch) or OTV (3-ch)")
    p.add_argument("--models",    nargs="+", choices=list(_MODEL_CHOICES), default=["unet_atv_v2"],
                   help="One or more architectures to train (default: unet_atv_v2)")
    p.add_argument("--loss",      choices=list(_LOSS_CHOICES), default="bce_dice")
    p.add_argument("--optimizer", choices=list(_OPT_CHOICES),  default="adam")
    p.add_argument("--lr",        type=float, default=5e-4)
    p.add_argument("--epochs",    type=int,   default=200)
    p.add_argument("--batch-size",type=int,   default=20)
    p.add_argument("--init-features", type=int, default=32, help="U-Net base channel count")
    p.add_argument("--seed",      type=int,   default=100)
    p.add_argument("--no-augment",action="store_true", help="Disable random flip augmentation")
    p.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.75],
                   help="Classification threshold(s) for test evaluation (default: 0.5 0.75)")
    p.add_argument("--run-name",  default=None,
                   help="Experiment name prefix (default: auto from date + loss + optimizer)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, test_ids: list, device: torch.device,
                   config: TrainingConfig, thresholds: list) -> dict:
    """Load best checkpoint and evaluate on test set.

    Returns dict: threshold → {accuracy, sensitivity, specificity, precision, f1, dice, n_samples}.
    """
    if not test_ids or not os.path.exists(model_path):
        return {}

    model = _build_model(config, device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    dataset = BoreholeDataset(list(test_ids), device=device)
    loader  = data.DataLoader(dataset, batch_size=1, shuffle=False)

    results = {}
    for thr in thresholds:
        tp = tn = fp = fn = 0
        dice_scores = []

        with torch.no_grad():
            for images, masks in loader:
                masks = _binarize_mask(masks).squeeze().cpu().numpy()
                images = images.to(device).float()
                pred_prob = model(images).squeeze().cpu().numpy()

                pred_bin = (pred_prob >= thr).astype(np.float32)
                m = masks.astype(np.float32)

                tp_i = int((pred_bin * m).sum())
                fp_i = int((pred_bin * (1 - m)).sum())
                fn_i = int(((1 - pred_bin) * m).sum())
                tn_i = int(((1 - pred_bin) * (1 - m)).sum())

                tp += tp_i; fp += fp_i; fn += fn_i; tn += tn_i

                denom = 2 * tp_i + fp_i + fn_i
                dice_scores.append((2 * tp_i / denom) if denom > 0 else 1.0)

        total = tp + tn + fp + fn
        results[thr] = {
            "accuracy":    100 * (tp + tn) / total       if total > 0          else 0.0,
            "sensitivity": 100 * tp / (tp + fn)          if (tp + fn) > 0      else 0.0,
            "specificity": 100 * tn / (tn + fp)          if (tn + fp) > 0      else 0.0,
            "precision":   100 * tp / (tp + fp)          if (tp + fp) > 0      else 0.0,
            "f1":          100 * 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) > 0   else 0.0,
            "dice_mean":   100 * float(np.mean(dice_scores)),
            "n_samples":   len(dataset),
        }
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def write_report(path: str, run_name: str, device: torch.device,
                 all_results: list):
    """Write markdown report for all trained models.

    all_results: list of dicts, one per model, with keys:
        model_name, config, train_results, eval_results, best_model_path
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    git = _git_hash()

    lines = [
        f"# Training Report — {run_name}",
        f"",
        f"**Date:** {now}  ",
        f"**Device:** {device}  ",
        f"**Git:** {git}  ",
        f"",
    ]

    for entry in all_results:
        cfg    = entry["config"]
        tr     = entry["train_results"]
        ev     = entry["eval_results"]
        mpath  = entry["best_model_path"]
        mname  = entry["model_name"]

        lines += [
            f"---",
            f"",
            f"## {mname}",
            f"",
            f"### Configuration",
            f"",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| Model type | `{cfg.model_type.value}` |",
            f"| Loss | `{cfg.loss_type.value}` |",
            f"| Optimizer | `{cfg.optimizer_type.value}` |",
            f"| Learning rate | {cfg.learning_rate} |",
            f"| LR decay (StepLR) | gamma={cfg.lr_gamma} every {cfg.lr_step_size} epochs |",
            f"| Max epochs | {cfg.max_epochs} |",
            f"| Batch size | {cfg.batch_size} (train) / {cfg.batch_size_val} (val) |",
            f"| Init features | {cfg.init_features} |",
            f"| Augmentation | {cfg.augment} |",
            f"| Seed | {cfg.seed} |",
            f"| Data dir | `{cfg.data_dir}` |",
            f"| Model dir | `{cfg.model_dir}` |",
            f"",
            f"### Training Summary",
            f"",
        ]

        train_losses = tr.get("training_losses", [])
        val_losses   = tr.get("validation_losses", [])
        best_val     = tr.get("best_validation_loss", float("inf"))
        n_test       = len(tr.get("test_ids", []))

        lines += [
            f"- Training samples: {len(tr.get('test_ids', []))+0} test held out",
            f"- Test set size: {n_test}",
            f"- Best validation loss: **{best_val:.4f}**",
            f"- Final training loss: {train_losses[-1]:.4f}" if train_losses else "- No training losses recorded",
            f"- Model saved: `{os.path.basename(mpath)}` (if it exists: {os.path.exists(mpath)})",
            f"",
            f"### Loss Curve (every 15 epochs)",
            f"",
            f"| Epoch | Train loss | Val loss |",
            f"|---|---|---|",
        ]

        stride = max(1, cfg.validate_every)
        val_idx = 0
        for i, tl in enumerate(train_losses):
            if i % stride == 0 or i == len(train_losses) - 1:
                vl_str = ""
                # val_losses are recorded every validate_every epochs (starting from epoch validate_every)
                epoch_no = i + 1
                if epoch_no > 0 and (epoch_no % cfg.validate_every == 0):
                    if val_idx < len(val_losses):
                        vl_str = f"{val_losses[val_idx]:.4f}"
                        val_idx += 1
                lines.append(f"| {i+1} | {tl:.4f} | {vl_str} |")

        lines.append("")

        if ev:
            lines += [
                f"### Test Set Evaluation",
                f"",
                f"| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | Dice (mean/img) |",
                f"|---|---|---|---|---|---|---|",
            ]
            for thr, m in ev.items():
                lines.append(
                    f"| {thr} "
                    f"| {m['accuracy']:.1f}% "
                    f"| {m['sensitivity']:.1f}% "
                    f"| {m['specificity']:.1f}% "
                    f"| {m['precision']:.1f}% "
                    f"| {m['f1']:.1f}% "
                    f"| {m['dice_mean']:.1f}% |"
                )
            lines.append(f"")
            lines.append(f"*n = {list(ev.values())[0]['n_samples']} test images*")
            lines.append("")
        else:
            lines += ["### Test Set Evaluation", "", "_No test data or model checkpoint not found._", ""]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir  = os.path.expanduser(args.data_dir)
    model_dir = os.path.expanduser(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = args.run_name or (
        f"{datetime.date.today().strftime('%Y%m%d')}"
        f"_{args.loss}_{args.optimizer}"
    )
    report_path = os.path.join(model_dir, f"{run_name}_report.md")

    all_results = []

    for model_key in args.models:
        model_type = _MODEL_CHOICES[model_key]
        data_type  = DataType.ATV if args.data_type == "atv" else DataType.OTV
        model_name = f"{run_name}_{model_key}"

        config = TrainingConfig(
            data_dir      = data_dir,
            model_dir     = model_dir,
            data_type     = data_type,
            model_type    = model_type,
            loss_type     = _LOSS_CHOICES[args.loss],
            optimizer_type= _OPT_CHOICES[args.optimizer],
            learning_rate = args.lr,
            max_epochs    = args.epochs,
            batch_size    = args.batch_size,
            batch_size_val= max(1, args.batch_size // 2),
            init_features = args.init_features,
            seed          = args.seed,
            augment       = not args.no_augment,
            model_name    = model_name,
        )

        print(f"\n{'='*60}")
        print(f"Training: {model_key}  loss={args.loss}  opt={args.optimizer}  lr={args.lr}")
        print(f"{'='*60}")

        train_results = train(config)

        best_model_path = os.path.join(model_dir, f"{model_name}_best.pt")
        eval_results    = evaluate_model(
            best_model_path,
            train_results.get("test_ids", []),
            device,
            config,
            args.threshold,
        )

        if eval_results:
            print(f"\nTest evaluation ({model_key}):")
            for thr, m in eval_results.items():
                print(f"  thr={thr}  acc={m['accuracy']:.1f}%  sens={m['sensitivity']:.1f}%"
                      f"  F1={m['f1']:.1f}%  Dice={m['dice_mean']:.1f}%")

        all_results.append({
            "model_name":      model_name,
            "config":          config,
            "train_results":   train_results,
            "eval_results":    eval_results,
            "best_model_path": best_model_path,
        })

    write_report(report_path, run_name, device, all_results)


if __name__ == "__main__":
    main()
