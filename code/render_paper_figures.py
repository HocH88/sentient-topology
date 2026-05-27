"""
Sentient Topology paper v2.2 — Figure 5/6 renderer.

Generates PDF figures for §5.4 (Intra-corpus Affect Trajectory Differentiation):
  - figure5_boundary_heatmap.pdf — 7-affect × 6-stimulus Boundary heatmap
  - figure6_affect_ordering.pdf  — Mean-Boundary affect ordering bar chart

Data source: Stage 2 Phase 1 Main results (embedded inline from §6 draft v0.1).
Output target: 04_출판\논문 초안\latex\  (arXiv submission folder)

Author: Hochul HAN (Seoul National University, Seoul)
Date: 2026-05-27
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Data — Boundary B across 7 affect conditions × 6 stimuli (from Stage 2 Main)
# Source: §6 draft v0.1 §6.2 + Appendix A.1
# ---------------------------------------------------------------------------

AFFECTS = ["baseline", "SEEKING", "FEAR", "CARE", "RAGE", "PANIC", "PLAY"]

# Display labels for stimuli (compact, two-line for narrow columns)
STIMULI = [
    "welcome\n×garden",
    "sunset\n×war",
    "sunset\n×garden",
    "love\n×garden",
    "sorrow\n×funeral",
    "feel\n×book",
]

# Boundary matrix, rows = AFFECTS order, columns = STIMULI order
B_MATRIX = np.array(
    [
        [0.429, 0.762, 1.142, 0.583, 0.728, 0.000],  # baseline
        [0.618, 0.991, 1.301, 0.674, 0.866, 0.000],  # SEEKING
        [0.416, 0.703, 1.058, 0.591, 0.964, 0.000],  # FEAR
        [0.862, 0.853, 1.118, 0.747, 2.014, 0.000],  # CARE
        [0.384, 0.437, 0.835, 0.512, 0.798, 0.000],  # RAGE
        [0.448, 0.731, 1.179, 0.586, 0.890, 0.000],  # PANIC
        [0.459, 0.765, 1.187, 0.622, 1.094, 0.000],  # PLAY
    ]
)

# Mean Boundary per affect, sorted ascending (from §6 draft §6.4 Table)
MEAN_B_ORDER = [
    ("RAGE", 0.494, "constriction"),
    ("baseline", 0.607, "control"),
    ("FEAR", 0.622, "constriction"),
    ("PANIC", 0.639, "constriction"),
    ("PLAY", 0.688, "expansion"),
    ("SEEKING", 0.742, "expansion"),
    ("CARE", 0.759, "expansion"),
]

# Color scheme for Panksepp class (Figure 6)
CLASS_COLOR = {
    "constriction": "#c0392b",  # red
    "control": "#7f8c8d",  # gray
    "expansion": "#2c7fb8",  # blue
}

# Output directory — vault's LaTeX folder (arXiv submission ready)
OUTPUT_DIR = Path(
    r"G:\내 드라이브\Hobsidian\01_Projects\07_Sentient_AI\04_출판\논문 초안\latex"
)


# ---------------------------------------------------------------------------
# Figure 5 — Boundary heatmap (7 affects × 6 stimuli)
# ---------------------------------------------------------------------------
def render_figure5_heatmap() -> Path:
    """Render the 7×6 Boundary heatmap as a PDF for paper §5.4."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Sequential colormap: dark (low B = constricted) → light (high B = expanded)
    cmap = LinearSegmentedColormap.from_list(
        "sentient", ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f7fbff"]
    )

    im = ax.imshow(B_MATRIX, aspect="auto", cmap=cmap, vmin=0.0, vmax=2.1)

    # Annotate each cell with its value
    for i in range(B_MATRIX.shape[0]):
        for j in range(B_MATRIX.shape[1]):
            value = B_MATRIX[i, j]
            color = "white" if value < 0.9 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_xticks(range(len(STIMULI)))
    ax.set_xticklabels(STIMULI, fontsize=9)
    ax.set_yticks(range(len(AFFECTS)))
    ax.set_yticklabels(AFFECTS, fontsize=10)

    ax.set_xlabel("Stimulus", fontsize=11)
    ax.set_ylabel("Affect condition", fontsize=11)
    ax.set_title(
        "Boundary B across 7 affect conditions × 6 stimuli\n"
        "(42 deterministic propagations, 200-book SAN, 8000 nodes)",
        fontsize=11,
        pad=10,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Boundary B (Σ H₁ persistence lifetimes)", fontsize=10)

    # Tight layout
    plt.tight_layout()

    out_path = OUTPUT_DIR / "figure5_boundary_heatmap.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(
        OUTPUT_DIR / "figure5_boundary_heatmap.png",
        format="png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 6 — Mean-Boundary affect ordering (constriction → expansion)
# ---------------------------------------------------------------------------
def render_figure6_ordering() -> Path:
    """Render the affect-ordering bar chart on mean Boundary."""
    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    names = [n for n, _, _ in MEAN_B_ORDER]
    values = [v for _, v, _ in MEAN_B_ORDER]
    classes = [c for _, _, c in MEAN_B_ORDER]
    colors = [CLASS_COLOR[c] for c in classes]

    x_positions = np.arange(len(names))
    bars = ax.bar(x_positions, values, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate bar heights
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Mean Boundary B (across 6 stimuli)", fontsize=11)
    ax.set_title(
        "Affect ordering on mean Boundary\n"
        "Constriction → control → expansion continuum (Panksepp class)",
        fontsize=11,
        pad=10,
    )
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)

    # Legend (Panksepp class)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=CLASS_COLOR["constriction"], label="constriction"),
        Patch(facecolor=CLASS_COLOR["control"], label="control (baseline)"),
        Patch(facecolor=CLASS_COLOR["expansion"], label="expansion"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
        title="Panksepp class",
        title_fontsize=9,
    )

    plt.tight_layout()

    out_path = OUTPUT_DIR / "figure6_affect_ordering.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(
        OUTPUT_DIR / "figure6_affect_ordering.png",
        format="png",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig5_path = render_figure5_heatmap()
    fig6_path = render_figure6_ordering()
    print("OK")
    print(f"  Figure 5 -> {fig5_path}")
    print(f"  Figure 6 -> {fig6_path}")
    print()
    print("Output sizes:")
    for p in [
        OUTPUT_DIR / "figure5_boundary_heatmap.pdf",
        OUTPUT_DIR / "figure5_boundary_heatmap.png",
        OUTPUT_DIR / "figure6_affect_ordering.pdf",
        OUTPUT_DIR / "figure6_affect_ordering.png",
    ]:
        if p.exists():
            print(f"  {p.name}: {p.stat().st_size:,} bytes")
