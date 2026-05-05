"""Generate poster-ready figures from the 20-run sweep results.

Usage (from project root):
    python code/make_plots.py

Outputs:
    figures/rank_curves.{png,pdf}     -- main story: rank robustness
    figures/pertask_r4_llama3.{png,pdf} -- per-task lift at low rank
    figures/poster_figure.{png,pdf}     -- combined panel for the poster
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Data (averages and per-task accuracies from results_v2/)
# ---------------------------------------------------------------------------
RANKS = [4, 8, 16, 32, 64]

# Avg over 7 tasks, in the order LoRA r=4..64, DoRA r=4..64
AVG = {
    "Llama-2-7B": {
        "LoRA": [65.64, 66.63, 66.81, 66.98, 67.33],
        "DoRA": [65.90, 66.49, 66.86, 66.27, 66.40],
    },
    "Llama-3-8B": {
        "LoRA": [69.92, 70.40, 71.17, 70.95, 70.99],
        "DoRA": [71.09, 71.03, 70.62, 70.82, 71.03],
    },
}

TASKS = ["BoolQ", "PIQA", "HellaSwag", "WinoGrande", "ARC-e", "ARC-c", "OBQA"]

PERTASK_R4 = {
    "Llama-3-8B": {
        "LoRA": [82.75, 81.45, 78.77, 74.11, 76.98, 50.77, 44.60],
        "DoRA": [83.39, 81.56, 78.63, 76.48, 80.18, 54.01, 43.40],
    },
    "Llama-2-7B": {
        "LoRA": [78.50, 79.05, 75.70, 71.82, 69.91, 40.87, 43.60],
        "DoRA": [77.74, 79.05, 75.55, 72.38, 70.71, 41.89, 44.00],
    },
}

# ---------------------------------------------------------------------------
# Style: clean, poster-friendly
# ---------------------------------------------------------------------------
LORA_C = "#3b6fb6"   # blue
DORA_C = "#e07a1f"   # orange
GAIN_C = "#2a8f3e"   # green
LOSS_C = "#c0392b"   # red

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.frameon": True,
    "legend.edgecolor": "0.7",
    "lines.linewidth": 2.6,
    "lines.markersize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, name, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path}")


def _plot_rank_curve(ax, model, show_legend=True, show_ylabel=True):
    lora = AVG[model]["LoRA"]
    dora = AVG[model]["DoRA"]
    ax.plot(RANKS, lora, "o-", color=LORA_C, label="LoRA")
    ax.plot(RANKS, dora, "s-", color=DORA_C, label="DoRA")

    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlabel("Rank $r$")
    if show_ylabel:
        ax.set_ylabel("Avg. accuracy (%)  —  7 commonsense tasks")
    ax.set_title(model)

    # annotate spread
    lora_spread = max(lora) - min(lora)
    dora_spread = max(dora) - min(dora)
    txt = (f"Range across ranks:\n"
           f"  LoRA: {lora_spread:.2f} pt\n"
           f"  DoRA: {dora_spread:.2f} pt")
    ax.text(0.04, 0.04, txt, transform=ax.transAxes, fontsize=11,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="white", edgecolor="0.8", alpha=0.95))

    if show_legend:
        ax.legend(loc="upper left")

    # tighten y-range so the curves use the panel
    ymin = min(min(lora), min(dora)) - 0.5
    ymax = max(max(lora), max(dora)) + 0.7
    ax.set_ylim(ymin, ymax)


def _plot_pertask_bars(ax, model, ylim=None, title_suffix=""):
    lora = PERTASK_R4[model]["LoRA"]
    dora = PERTASK_R4[model]["DoRA"]
    x = np.arange(len(TASKS))
    width = 0.36
    ax.bar(x - width / 2, lora, width, label="LoRA  $r{=}4$", color=LORA_C, edgecolor="white")
    ax.bar(x + width / 2, dora, width, label="DoRA  $r{=}4$", color=DORA_C, edgecolor="white")

    # delta annotations above each pair
    for i, (l, d) in enumerate(zip(lora, dora)):
        delta = d - l
        sign = "+" if delta > 0 else ""
        color = GAIN_C if delta > 0 else LOSS_C
        h = max(l, d)
        ax.text(i, h + 1.1, f"{sign}{delta:.2f}",
                ha="center", va="bottom",
                fontsize=11, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-task accuracy at $r{{=}}4$ on {model}{title_suffix}")
    ax.legend(loc="upper right")
    if ylim is None:
        ymin = min(min(lora), min(dora)) - 4
        ymax = max(max(lora), max(dora)) + 5
        ylim = (ymin, ymax)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")
    ax.grid(False, axis="x")


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def make_rank_curves(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, model in zip(axes, ["Llama-2-7B", "Llama-3-8B"]):
        _plot_rank_curve(ax, model, show_ylabel=(ax is axes[0]))
    fig.suptitle("DoRA's accuracy is flatter across ranks than LoRA's",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "rank_curves", outdir)
    plt.close(fig)


def make_pertask_r4(outdir):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_pertask_bars(ax, "Llama-3-8B",
                       title_suffix="  —  DoRA wins on the hardest tasks")
    fig.tight_layout()
    _save(fig, "pertask_r4_llama3", outdir)
    plt.close(fig)


def make_poster_figure(outdir):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.25)

    ax_l2 = fig.add_subplot(gs[0, 0])
    ax_l3 = fig.add_subplot(gs[0, 1])
    _plot_rank_curve(ax_l2, "Llama-2-7B", show_legend=True, show_ylabel=True)
    _plot_rank_curve(ax_l3, "Llama-3-8B", show_legend=True, show_ylabel=True)

    ax_pt = fig.add_subplot(gs[1, :])
    _plot_pertask_bars(ax_pt, "Llama-3-8B",
                       title_suffix="  —  the regime where DoRA's gap is largest")

    fig.suptitle("DoRA vs. LoRA on Commonsense Reasoning: rank robustness "
                 "& low-rank lift",
                 fontsize=19, fontweight="bold", y=0.995)

    _save(fig, "poster_figure", outdir)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    outdir = Path(__file__).resolve().parent.parent / "figures"
    print(f"Writing figures to: {outdir}")
    make_rank_curves(outdir)
    make_pertask_r4(outdir)
    make_poster_figure(outdir)
    print("Done.")


if __name__ == "__main__":
    main()
