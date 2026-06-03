# DeepLogger Wiki

Living knowledge base for the DeepLogger project and source material for the scientific manuscript.

**Citation style:** `[@key]` (Pandoc-compatible → `\cite{key}` in LaTeX).  
**BibTeX file:** [`references.bib`](references.bib) — import directly into LaTeX/Overleaf.  
**Convert to LaTeX:** `pandoc <file>.md --bibliography=references.bib -o <file>.tex`

---

## Wiki Pages

| Page | Contents | Paper section |
|---|---|---|
| [Background](background.md) | Motivation, problem statement, related work | §1 Introduction, §2 Related Work |
| [Dataset](dataset.md) | Bedretto Lab, ATV/OTV instruments, data stats, preprocessing | §3 Data |
| [Methods](methods.md) | Architectures, loss functions, training strategy | §4 Methods |
| [Results](results.md) | Evaluation metrics, model comparison table | §5 Results |
| [References](references.md) | Formatted bibliography (all cited works) | §8 References |
| [`references.bib`](references.bib) | BibTeX source | — |

---

## Paper Outline & Status

> Draft status: ◻ not started · ◑ in progress · ✓ ready

| § | Title | Status | Key claims | Figures |
|---|---|---|---|---|
| 1 | Introduction | ◑ | fracture detection bottleneck; ML gaps; contributions | — |
| 2 | Related Work | ◑ | prior ML on borehole images; U-Net family; attention gates | — |
| 3 | Data | ◑ | Bedretto Lab; 16 boreholes; 1709 ATV snippets; SVD prep | Fig. 1 (site map), Fig. 2 (ATV/OTV examples) |
| 4 | Methods | ◑ | UNetATV v2 + AttentionUNetATV; BCE+Dice loss | Fig. 3 (architecture diagram) |
| 5 | Results | ◻ | training curves; test metrics; comparison table | Fig. 4 (loss curves), Fig. 5 (metric table), Fig. 6 (prediction overlay) |
| 6 | Discussion | ◻ | imbalance problem; SVD dependency; fine-tuning | — |
| 7 | Conclusion | ◻ | — | — |
| 8 | References | ◑ | see references.md | — |

---

## Figures Inventory

| ID | Description | Source | Status |
|---|---|---|---|
| Fig. 1 | Bedretto Lab site map + borehole layout | [@plenkers2023] | need to request/redraw |
| Fig. 2 | Example ATV and OTV log snippets with fracture labels | this work | need to generate from data |
| Fig. 3 | UNetATV v2 and AttentionUNetATV block diagrams | `docs/architecture_diagram.png` | ✓ exists |
| Fig. 4 | Training + validation loss curves | output of `run_training.py` report | ◻ pending GPU run |
| Fig. 5 | Sensitivity / F1 / Dice comparison table | output of `run_training.py` report | ◻ pending GPU run |
| Fig. 6 | Qualitative prediction overlay on test borehole | inferencer GUI | ◻ pending GPU run |

---

## Key Implementation Files

| File | Role |
|---|---|
| `deeplogger/model_architectures_v2.py` | UNetATV, AttentionUNetATV, AttentionGate |
| `deeplogger/loss_functions.py` | BCEDiceLoss, DiceLoss |
| `deeplogger/train.py` | `train()`, `finetune()`, `_build_model()` |
| `deeplogger/config.py` | TrainingConfig, ModelType, LossType enums |
| `scripts/run_training.py` | CLI training + evaluation + markdown report |
| `docs/DESIGN.md` | Full technical reference (implementation details) |

---

## Open Questions for the Paper

- [ ] What is the appropriate stratified test split? (borehole-level, not random — see [methods.md](methods.md))
- [ ] Does AttentionUNetATV outperform UNetATV v2 on ATV data? → answer after GPU training run
- [ ] Should inference pipeline apply SVD via GUI or auto-apply with fixed k=3?
- [ ] Is the aperture extraction method validated enough for inclusion as a Methods subsection?
- [ ] Which boreholes are SB vs CB vs MB? Check [@plenkers2023] for borehole naming.
