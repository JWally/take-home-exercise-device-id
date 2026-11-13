"""Render the matched frontier and clearly labelled historical figures into docs/img/.

The historical figures retain docs/figures_data.json and the ledger values below.

.venv/bin/python research/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "docs"
IMG = DOCS / "img"
D = json.loads((DOCS / "figures_data.json").read_text())

from chart_style import configure_style, header  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET, RED = "#244b68", "#aa7c30", "#3d837b", "#746b8e", "#b3262e"
INK, INK2, MUTED, GRID = "#202a35", "#58616a", "#8b9092", "#d8d9d5"
PAPER = "#f5f4ef"

configure_style()


def historical_figure(title, subtitle, *, height=5.9, left=0.29):
    """Give the retained exploratory charts the same editorial frame."""
    fig, ax = plt.subplots(figsize=(10, height))
    fig.subplots_adjust(left=left, right=0.94, top=0.76, bottom=0.21)
    header(fig, title, subtitle, eyebrow="HISTORICAL EXPLORATION")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    return fig, ax


def save(fig, name, note=""):
    IMG.mkdir(exist_ok=True)
    fig.text(
        0.075,
        0.065,
        "Source: original experiment records in docs/figures_data.json and research/make_figures.py.",
        fontsize=8,
        color=INK2,
    )
    fig.text(
        0.075,
        0.032,
        note
        or "Historical exploratory results; evaluation separation was not independently established.",
        fontsize=8,
        color=INK2,
    )
    fig.savefig(IMG / name, dpi=180, facecolor=PAPER)
    plt.close(fig)
    print("wrote", IMG / name)


def interp(frontier, fpr):
    x, y = np.array(frontier).T
    return float(np.interp(fpr, x, y))


def fig_frontier():
    """The primary comparison comes from the matched three-fold evidence."""
    from frontier_charts import render_all

    render_all()


def fig_ledger():
    """Historical deltas against an interpolated original size-aware curve."""
    sa = D["frontier"]["size_aware"]
    slope = (
        interp(sa, 1.41) - interp(sa, 1.0)
    ) / 0.41  # recall points per FPR point near the default

    def d(r, f):
        return r - interp(sa, f)

    items = [
        (
            "Plain connected components",
            np.mean([d(r, f) for f, r in D["frontier"]["components"] if 0.5 < f < 2]),
        ),
        (
            "Learned gate, fixed threshold",
            np.mean([d(r, f) for f, r in D["frontier"]["fixed_gate"] if 0.5 < f < 2]),
        ),
        ("Detach weakly attached sessions", d(87.58, 0.94)),
        ("Lower edge floor (0.05)", np.mean([d(88.27, 1.285), d(89.67, 2.20)])),
        ("Size-term exponent 0.7", d(86.90, 0.842)),
        (
            "Expected-purity merge rule",
            np.mean([d(88.00, 0.978), d(89.49, 1.783), d(86.85, 0.745)]),
        ),
        (
            "Gate trained with pair/session weights",
            np.mean([d(88.02, 0.92), d(88.21, 1.21), d(87.33, 0.91)]),
        ),
        ("Bigger gate (900 trees, depth 8)", np.mean([d(88.90, 1.383), d(89.76, 1.886)])),
        ("Size-term exponent 1.3 / 1.5", np.mean([d(89.28, 1.546), d(89.00, 1.418)])),
        ("Drop labeled sessions from graph", np.mean([d(88.20, 1.01), d(89.49, 1.63)])),
        (
            "Per-cohort thresholds (original CV)",
            np.mean(
                [
                    (89.18 - 89.22) - slope * 0.15,
                    (87.66 - 87.19) - slope * 0.21,
                    (87.80 - 87.61) - slope * 0.17,
                ]
            ),
        ),
        (
            "Android drift features (vs same seed)",
            np.mean(
                [
                    (88.49 - 88.41) - slope * (1.156 - 1.166),
                    (90.10 - 89.94) - slope * (2.045 - 1.919),
                ]
            ),
        ),
    ]
    items.sort(key=lambda t: t[1])
    fig, ax = historical_figure(
        "Small changes, uncertain gains",
        "Original merge experiments · recall-point change against the interpolated size-aware curve",
        height=6.9,
        left=0.39,
    )
    names, vals = zip(*items, strict=True)
    colors = [RED if v < -0.25 else (MUTED if abs(v) <= 0.25 else AQUA) for v in vals]
    y = np.arange(len(items))
    ax.axvspan(-0.25, 0.25, color=GRID, alpha=0.45, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.scatter(vals, y, color=colors, s=38, zorder=3)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", labelsize=9)
    for i, v in enumerate(vals):
        ax.text(v + 0.045, i, f"{v:+.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Recall difference, percentage points", labelpad=10)
    ax.set_xlim(-1.4, 0.7)
    ax.set_ylim(-0.65, len(items) - 0.35)
    save(
        fig,
        "ledger_merge.png",
        "Historical estimates, not matched refits. Shading: ±0.25-point reference band, not a confidence interval.",
    )


def fig_ledger_scoring():
    """CV pairwise F1 of the earlier, F1-era experiments on the pair scorer and partitioner."""
    base = 92.10
    items = [
        ("Correlation clustering", 92.18),
        ("Average linkage", 92.00),
        ("Neighborhood stacking features", 92.17),
        ("Crowding / joint-key features", 92.02),
        ("Competitor features", 92.14),
        ("UA / viewport / battery / rarity features", 92.35),
        ("Cluster-pair merge model", 92.42),
        ("Gate retrained on own mistakes (DAgger)", 92.68),
        ("Learned merge gate", 92.82),
        ("Learned gate + better pair features", 92.99),
    ]
    fig, ax = historical_figure(
        "Earlier experiments improved pairwise F1",
        "Original scoring and clustering experiments · baseline F1: 92.10%",
        height=6.2,
        left=0.40,
    )
    names, vals = zip(*items, strict=True)
    delta = np.array(vals) - base
    y = np.arange(len(items))
    colors = [AQUA if v > 0.25 else MUTED for v in delta]
    ax.axvspan(-0.25, 0.25, color=GRID, alpha=0.45, zorder=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.scatter(delta, y, color=colors, s=38, zorder=3)
    ax.set_yticks(y, names)
    ax.tick_params(axis="y", labelsize=9)
    for i, (v, raw) in enumerate(zip(delta, vals, strict=True)):
        ax.text(v + 0.04, i, f"{raw:.2f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("F1 difference, percentage points · labels show absolute F1", labelpad=10)
    ax.set_xlim(-0.4, 1.2)
    ax.set_ylim(-0.65, len(items) - 0.35)
    save(
        fig,
        "ledger_scoring.png",
        "Historical exploratory scores; not the grader's recall/error objective. Shading is a reference band only.",
    )


def fig_cohorts():
    C = sorted(D["cohorts"], key=lambda c: c["recall"])
    four = sum(c["share_of_misses"] for c in C if c["share_of_misses"] >= 10)
    fig, ax = historical_figure(
        "Mobile browsers account for most missed links",
        f"Four cohorts account for {four:.0f}% of misses in the original default-setting analysis",
        height=6.1,
        left=0.23,
    )
    names = [
        c["cohort"]
        .replace("chrome", "Chrome")
        .replace("safari", "Safari")
        .replace("firefox", "Firefox")
        .replace("edge", "Edge")
        .replace("samsung", "Samsung")
        for c in C
    ]
    y = np.arange(len(C))
    colors = [ORANGE if c["share_of_misses"] >= 10 else BLUE for c in C]
    ax.scatter([c["recall"] for c in C], y, color=colors, s=40, zorder=3)
    ax.set_yticks(y, names)
    for i, c in enumerate(C):
        ax.text(c["recall"] + 0.7, i, f"{c['recall']:.1f}", va="center", fontsize=9, color=INK)
        ax.text(
            111, i, f"{c['share_of_misses']:.0f}%", va="center", ha="right", fontsize=9, color=INK2
        )
    ax.text(111, len(C) - 0.03, "Share of misses", ha="right", fontsize=9, color=INK2)
    ax.set_xlim(65, 112)
    ax.set_xticks(range(70, 101, 10))
    ax.set_ylim(-0.65, len(C) - 0.1)
    ax.set_xlabel("Recall, % of same-device pairs linked", labelpad=10)
    save(fig, "cohorts.png")


def fig_stability():
    S = D["stability"]
    labels = {
        "ip": "IP",
        "asn": "ASN (network)",
        "user_agent": "User agent",
        "languages": "Languages",
        "text_shaping_hash": "Text shaping",
        "media_speech_voice_hash": "Speech voices",
    }
    keys = list(labels)
    y = np.arange(len(keys))
    fig, ax = historical_figure(
        "Android signals changed more often",
        "Devices with multiple sessions · share whose signal changed between visits, %",
        height=5.5,
        left=0.23,
    )
    android = [S["Android chrome"][k] for k in keys]
    windows = [S["Windows chrome"][k] for k in keys]
    ax.hlines(y, windows, android, color=GRID, linewidth=2, zorder=2)
    ax.scatter(android, y, color=ORANGE, label="Android Chrome", s=45, zorder=3)
    ax.scatter(windows, y, color=BLUE, label="Windows Chrome", s=45, zorder=3)
    ax.set_yticks(y, [labels[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(-2, 60)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("Devices with a changing signal, %", labelpad=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=2, fontsize=9)
    save(
        fig,
        "stability.png",
        "Historical descriptive analysis of the original dataset; not a held-out model benchmark.",
    )


def fig_calibration():
    cal = [c for c in D["calibration"] if c["sessions"] >= 50]
    fig, ax = historical_figure(
        "Reported confidence tracked observed correctness",
        "Original calibration bins · descriptive evidence, not a calibration guarantee",
        height=6.3,
        left=0.15,
    )
    ax.set_position([0.15, 0.19, 0.66, 0.55])
    ax.plot([85, 100], [85, 100], color=MUTED, ls=(0, (3, 4)), lw=1, label="Perfect agreement")
    p = np.array([c["predicted"] for c in cal]) * 100
    o = np.array([c["observed"] for c in cal]) * 100
    n = np.array([c["sessions"] for c in cal])
    ax.scatter(p, o, s=25 + 150 * n / n.max(), color=BLUE, edgecolors=PAPER, lw=1, zorder=4)
    ax.set_xlim(85, 100.5)
    ax.set_ylim(85, 100.5)
    ax.set_xticks([85, 90, 95, 100])
    ax.set_yticks([85, 90, 95, 100])
    ax.set_xlabel("Predicted link confidence, %", labelpad=10)
    ax.set_ylabel("Observed correct links, %", labelpad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.03, 0.83, "Dot area scales with sessions", transform=ax.transAxes, fontsize=9, color=INK2
    )
    save(
        fig,
        "calibration.png",
        "Historical exploratory evaluation; separation not independently established. The 26-session tail bin is omitted.",
    )


def fig_split_noise():
    x = np.array(D["split_noise_f1"])
    fig, ax = historical_figure(
        "The chosen split moves the score",
        "One model · pairwise F1 across 300 random subsets of 2,033 devices",
        height=5.2,
        left=0.13,
    )
    ax.hist(x, bins=30, color=BLUE, edgecolor=PAPER, linewidth=0.6, zorder=3)
    ax.axvline(x.mean(), color=RED, lw=1.4)
    ax.text(
        0.03,
        0.94,
        f"Mean {x.mean():.1f}%  ·  standard deviation {x.std():.1f} points",
        transform=ax.transAxes,
        va="top",
        color=INK,
        fontsize=10,
    )
    ax.set_xlabel("Pairwise F1, %", labelpad=10)
    ax.set_ylabel("Subsets", labelpad=10)
    save(
        fig,
        "split_noise.png",
        "Historical resampling of a fixed dataset; subsets overlap and are not 300 independent test sets.",
    )


def fig_old_split():
    rows = [
        ("Frozen baseline\nCatBoost ensemble", 91.42, 94.44),
        ("History model", 92.93, 94.42),
        ("Signed history\nBest inspected-test result", 92.85, 94.87),
        ("Main's earlier F1 setting", 92.51, 92.66),
    ]
    fig, ax = historical_figure(
        "The original splits told a different story",
        "Earlier model comparisons · pairwise F1 with all 65,478 sessions as distractors",
        height=5.6,
        left=0.29,
    )
    y = np.arange(len(rows))
    ax.hlines(y, [r[1] for r in rows], [r[2] for r in rows], color=GRID, linewidth=2, zorder=2)
    ax.scatter([r[1] for r in rows], y, color=BLUE, label="Original validation", s=45, zorder=3)
    ax.scatter(
        [r[2] for r in rows], y, color=ORANGE, label="Inspected original test", s=45, zorder=3
    )
    for i, r in enumerate(rows):
        ax.annotate(
            f"{r[1]:.2f}",
            (r[1], i),
            (0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=BLUE,
        )
        ax.annotate(
            f"{r[2]:.2f}",
            (r[2], i),
            (0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=ORANGE,
        )
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.45, -0.65)
    ax.set_xlim(90.8, 95.4)
    ax.set_xticks(range(91, 96))
    ax.set_xlabel("Pairwise F1, %", labelpad=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=2, fontsize=9)
    save(
        fig,
        "old_split.png",
        "Both splits have been inspected. Historical scores are not an untouched-test comparison or the current objective.",
    )


if __name__ == "__main__":
    fig_frontier()
    fig_ledger()
    fig_ledger_scoring()
    fig_cohorts()
    fig_stability()
    fig_calibration()
    fig_split_noise()
    fig_old_split()
