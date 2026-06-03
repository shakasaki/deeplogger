# Results

*Paper section: §5 Results*

**Status: ◻ pending GPU training run.**

This page will be populated after running `scripts/run_training.py` on the GPU server.
See [INDEX.md](INDEX.md) for the training command.

---

## 1. Thesis Baseline (Perritaz 2024, ATV)

From [@perritaz2024, Table 4] — best ATV model: UNetATV v1, BCE loss, Adam, epoch 75.
Evaluated on boreholes SB 2.3 and SB 3.1.

| Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---|---|---|---|---|
| 0.50 | 99 % | 27 % | ~100 % | ~2 % | — |
| 0.75 | ~90 % ± 12 % | ~2 % ± 4 % | ~100 % | ~2 % ± 3–9 % | 18 % |

**Notes:**
- High accuracy is misleading (dominated by background class)
- Mean sensitivity 81–90 %, F1 13–32 % across test images and thresholds [@perritaz2024, §5.1]
- Low precision (2 %) is expected under <1 % foreground fraction [@li2019overfitting]
- Results fluctuate significantly with threshold — post-processing (sinusoidal fitting) is critical

---

## 2. Planned Experiments

| Run ID | Model | Loss | Optimizer | LR | Epochs | Expected |
|---|---|---|---|---|---|---|
| `atv_bce_adam_baseline` | UNetATV v1 | BCE | Adam | 0.1 | 200 | Reproduce thesis |
| `atv_bce_dice_v2` | UNetATV v2 | BCE+Dice | Adam | 5e-4 | 200 | Improve F1 over baseline |
| `atv_bce_dice_attn` | AttentionUNetATV | BCE+Dice | Adam | 5e-4 | 200 | Best expected |

Command (GPU server):

```bash
python scripts/run_training.py \
    --data-dir ~/DATA/.../atv_data_label/ \
    --models unet_atv_v2 attention_unet_atv \
    --loss bce_dice --optimizer adam --lr 5e-4 \
    --epochs 200 --run-name atv_bce_dice_adam
```

Report written to: `models/atv_bce_dice_adam_report.md`

---

## 3. Results Table

*To be filled after training.*

| Model | Threshold | Sensitivity | F1 | Dice (mean/img) | Best epoch |
|---|---|---|---|---|---|
| UNetATV v1 (BCE, Adam, lr=0.1) | 0.50 | — | — | — | — |
| UNetATV v2 (BCE+Dice, Adam, lr=5e-4) | 0.50 | — | — | — | — |
| AttentionUNetATV (BCE+Dice, Adam, lr=5e-4) | 0.50 | — | — | — | — |
| UNetATV v1 (BCE, Adam, lr=0.1) | 0.75 | — | — | — | — |
| UNetATV v2 (BCE+Dice, Adam, lr=5e-4) | 0.75 | — | — | — | — |
| AttentionUNetATV (BCE+Dice, Adam, lr=5e-4) | 0.75 | — | — | — | — |

---

## 4. Notes on Interpretation

- **Sensitivity vs precision tradeoff:** increasing the classification threshold increases
  precision but reduces sensitivity. Post-processing (connected component filtering,
  sinusoidal fitting) can recover precision without sacrificing recall.
- **Per-image Dice** is more informative than batch-level Dice for variable-size images.
- **Borehole-stratified test split** (TODO) will give a more realistic estimate of
  generalisation to new boreholes.

---

## References (this page)

`[@perritaz2024]`, `[@li2019overfitting]`

Full BibTeX: [`references.bib`](references.bib)
