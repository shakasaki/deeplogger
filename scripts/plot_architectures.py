"""Plot schematic block diagrams for UNetATV and AttentionUNetATV.

Run directly to save ``docs/architecture_diagram.png``:

    python scripts/plot_architectures.py
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
BLOCK_W = 1.4       # width of a feature block rectangle
BLOCK_H = 0.55      # height of a feature block rectangle
COL_GAP = 0.55      # horizontal gap between encoder and decoder columns
LEVEL_H = 1.1       # vertical distance between levels
ARROW_KW = dict(arrowstyle="-|>", color="#333333", lw=1.2,
                mutation_scale=10, connectionstyle="arc3,rad=0.")

# Colour scheme
C_ENC    = "#4C72B0"   # encoder blocks
C_BN     = "#DD8452"   # bottleneck
C_DEC    = "#55A868"   # decoder blocks
C_ATT    = "#C44E52"   # attention gate marker
C_DROP   = "#8172B2"   # dropout marker
C_SKIP   = "#aaaaaa"   # skip-connection arrows
C_NORM   = "#64B5CD"   # normalization note

FONT_SM  = 7
FONT_MD  = 8.5


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rect(ax, x, y, w, h, color, label, sublabel="", alpha=0.85):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04", linewidth=0.8,
        edgecolor="white", facecolor=color, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.03, label, ha="center", va="center",
            fontsize=FONT_MD, fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel, ha="center", va="center",
                fontsize=FONT_SM - 0.5, color="white", alpha=0.9, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="#333333", style="->", lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, mutation_scale=9))


def _skip(ax, x_enc, y_enc, x_dec, y_dec, label="", att=False, dash=False):
    """Draw a horizontal skip-connection arrow."""
    ls = "--" if dash else "-"
    color = C_ATT if att else C_SKIP
    lw = 1.5 if att else 1.0
    ax.annotate("", xy=(x_dec - BLOCK_W / 2, y_dec),
                xytext=(x_enc + BLOCK_W / 2, y_enc),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, linestyle=ls, mutation_scale=9,
                                connectionstyle="arc3,rad=0."))
    if label:
        mx = (x_enc + BLOCK_W / 2 + x_dec - BLOCK_W / 2) / 2
        my = (y_enc + y_dec) / 2
        ax.text(mx, my + 0.14, label, ha="center", va="bottom",
                fontsize=FONT_SM, color=color, zorder=5)


def _badge(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=FONT_SM - 0.5,
            color="white", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color,
                      edgecolor="white", lw=0.6, alpha=0.95))


# ---------------------------------------------------------------------------
# Architecture spec (shared between both models)
# ---------------------------------------------------------------------------
# Each entry: (label, ch_out, is_encoder)
# Levels 0..3 = enc1..enc4; level 4 = bottleneck; levels 5..8 = dec4..dec1

_LEVELS = [
    # encoder
    ("enc1",  32,  "enc"),
    ("enc2",  64,  "enc"),
    ("enc3",  128, "enc"),
    ("enc4",  256, "enc"),
    # bottleneck
    ("bottle", 512, "bn"),
    # decoder
    ("dec4",  256, "dec"),
    ("dec3",  128, "dec"),
    ("dec2",  64,  "dec"),
    ("dec1",  32,  "dec"),
]

_N_ENC = 4
_N_DEC = 4


def _draw_architecture(ax, title: str, attention: bool, dropout: bool):
    """Draw one architecture diagram on *ax*."""
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.6, 5.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # Column x-positions
    x_enc = 2.5
    x_dec = 7.5

    # Encoder levels go *downward* from top
    enc_ys = [4.5 - i * LEVEL_H for i in range(_N_ENC)]   # y per encoder level
    bn_y   = 4.5 - _N_ENC * LEVEL_H                        # bottleneck y
    dec_ys = [bn_y + (i + 1) * LEVEL_H for i in range(_N_DEC)]  # decoder goes back up

    # --- Normalisation note (inside first encoder) ---
    norm_y = enc_ys[0] + 0.55
    _rect(ax, x_enc, norm_y, BLOCK_W * 0.9, BLOCK_H * 0.7, C_NORM,
          "Norm", "per-sample\nmin–max")

    # --- Input arrow ---
    _arrow(ax, x_enc, norm_y + 0.42, x_enc, norm_y + 0.25, color="#333")
    ax.text(x_enc, norm_y + 0.55, "input (1, H, W)", ha="center",
            va="bottom", fontsize=FONT_SM, color="#444")

    # --- Encoder blocks ---
    drop_levels = {2, 3} if dropout else set()  # enc3, enc4 get dropout

    for i, (label, ch, kind) in enumerate(_LEVELS[:_N_ENC]):
        y = enc_ys[i]
        color = C_ENC
        _rect(ax, x_enc, y, BLOCK_W, BLOCK_H, color,
              f"{label}", f"{ch} ch")
        if i in drop_levels:
            _badge(ax, x_enc + BLOCK_W / 2 + 0.22, y + 0.12, "D", C_DROP)
        # Pool arrow going down (except after enc4 → different for v1/v2)
        if i < _N_ENC - 1:
            _arrow(ax, x_enc, y - BLOCK_H / 2,
                   x_enc, enc_ys[i + 1] + BLOCK_H / 2,
                   color="#555")
            ax.text(x_enc - 0.55, (y + enc_ys[i + 1]) / 2,
                    "pool\n2×2", ha="center", va="center",
                    fontsize=FONT_SM - 1, color="#666")
        else:
            # pool4 label depends on architecture
            pool4_label = "pool\n2×2" if attention else "pool\n3×3/s1"
            ax.text(x_enc - 0.55, (y + bn_y) / 2,
                    pool4_label, ha="center", va="center",
                    fontsize=FONT_SM - 1,
                    color="#c0392b" if not attention else "#666")
            _arrow(ax, x_enc, y - BLOCK_H / 2,
                   x_enc, bn_y + BLOCK_H / 2, color="#555")

    # --- Bottleneck ---
    _rect(ax, 5.0, bn_y, BLOCK_W * 1.1, BLOCK_H, C_BN, "bottleneck", "512 ch")
    if dropout:
        _badge(ax, 5.0 + BLOCK_W * 1.1 / 2 + 0.22, bn_y + 0.12, "D", C_DROP)
    _arrow(ax, x_enc + BLOCK_W / 2, bn_y, 5.0 - BLOCK_W * 1.1 / 2, bn_y,
           color="#555")
    _arrow(ax, 5.0 + BLOCK_W * 1.1 / 2, bn_y, x_dec - BLOCK_W / 2, bn_y,
           color="#555")

    # --- Decoder blocks ---
    for i, (label, ch, kind) in enumerate(_LEVELS[_N_ENC + 1:]):
        y = dec_ys[i]
        _rect(ax, x_dec, y, BLOCK_W, BLOCK_H, C_DEC, f"{label}", f"{ch} ch")

        # upconv arrow
        prev_y = dec_ys[i - 1] if i > 0 else bn_y
        _arrow(ax, x_dec, prev_y - BLOCK_H / 2 if i == 0 else prev_y - BLOCK_H / 2,
               x_dec, y + BLOCK_H / 2, color="#555")
        ax.text(x_dec + 0.55, (prev_y + y) / 2,
                "upconv\n2×2", ha="center", va="center",
                fontsize=FONT_SM - 1, color="#666")

        # Skip connection from matching encoder level (enc4→dec4, ..., enc1→dec1)
        enc_i = _N_ENC - 1 - i
        enc_y = enc_ys[enc_i]
        if attention:
            _skip(ax, x_enc, enc_y, x_dec, y,
                  label="att", att=True)
        else:
            _skip(ax, x_enc, enc_y, x_dec, y, dash=(enc_i < 2))

    # --- Output arrow ---
    top_dec_y = dec_ys[-1]
    _arrow(ax, x_dec, top_dec_y + BLOCK_H / 2,
           x_dec, top_dec_y + BLOCK_H / 2 + 0.45, color="#333")
    ax.text(x_dec, top_dec_y + BLOCK_H / 2 + 0.55,
            "Conv1×1 → sigmoid\noutput (H, W)",
            ha="center", va="bottom", fontsize=FONT_SM, color="#444")

    # --- Legend ---
    legend_x, legend_y = 0.1, 1.0
    patches = [
        mpatches.Patch(color=C_ENC,  label="Conv block (2× Conv3+BN+ReLU)"),
        mpatches.Patch(color=C_BN,   label="Bottleneck"),
        mpatches.Patch(color=C_DEC,  label="Decoder block"),
        mpatches.Patch(color=C_NORM, label="Per-sample normalisation"),
    ]
    if attention:
        patches += [
            mpatches.Patch(color=C_ATT,  label="Attention gate (skip)"),
            mpatches.Patch(color=C_DROP, label="Spatial dropout"),
        ]
    ax.legend(handles=patches, loc="lower left", fontsize=FONT_SM - 0.5,
              framealpha=0.8, edgecolor="#ccc")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_architectures(save_path: str | None = None) -> plt.Figure:
    """Generate side-by-side block diagrams for both v2 architectures.

    Args:
        save_path: Optional file path to save the figure (PNG).

    Returns:
        The matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("DeepLogger ATV U-Net Architectures", fontsize=13,
                 fontweight="bold", y=1.01)

    _draw_architecture(axes[0], "UNetATV  (Perritaz 2024 + normalisation)",
                       attention=False, dropout=False)
    _draw_architecture(axes[1], "AttentionUNetATV  (attention + dropout)",
                       attention=True, dropout=True)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    return fig


if __name__ == "__main__":
    import pathlib
    repo = pathlib.Path(__file__).parent.parent
    plot_architectures(str(repo / "docs" / "architecture_diagram.png"))
    plt.show()
