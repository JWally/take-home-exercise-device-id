"""A restrained editorial style for reproducible research figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

PAPER = "#f5f4ef"
INK = "#202a35"
MUTED = "#60676d"
GRID = "#d8d9d5"
RED = "#b3262e"
COLORS = {
    "current_cost_guard": "#244b68",
    "nested_cost_gate": "#3d837b",
    "cost_size_utility": "#aa7c30",
    "main_cost_hybrid": "#746b8e",
    "main_refit": RED,
}


def configure_style():
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.axisbelow": True,
            "axes.grid": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.major.pad": 8,
            "ytick.major.pad": 8,
            "legend.frameon": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "device-id-frontier-v1",
            "savefig.dpi": 190,
        }
    )


def header(fig, title, subtitle="", eyebrow="DEVICE LINKING"):
    fig.add_artist(
        Rectangle(
            (0.075, 0.957), 0.065, 0.008, transform=fig.transFigure, facecolor=RED, edgecolor="none"
        )
    )
    fig.text(0.075, 0.918, eyebrow, fontsize=9, color=MUTED, weight="bold")
    fig.text(0.075, 0.86, title, fontsize=23, fontfamily="DejaVu Serif")
    if subtitle:
        fig.text(0.075, 0.812, subtitle, fontsize=11, color=MUTED)
