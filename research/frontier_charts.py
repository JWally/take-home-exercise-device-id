"""Render matched development frontiers and frozen validation operating points.

Only committed aggregate counts are read. No raw sessions, labels, model fitting
or inferred operating points are needed. Run: python research/frontier_charts.py
"""

import csv
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_style import COLORS, GRID, MUTED, PAPER, configure_style, header  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/frontier_data.json"
IMG = ROOT / "docs/img"
LABELS = {
    "current_cost_guard": "XGBoost + merge rules",
    "nested_cost_gate": "XGBoost + merge model",
    "cost_size_utility": "XGBoost + merge costs",
    "main_cost_hybrid": "CatBoost + XGBoost blend",
    "main_refit": "CatBoost + merge model (refit)",
}
CAPS = [Fraction(n, 100) for n in range(1, 6)]
FOLD_SESSIONS = [6882, 6495, 6358]
FOLD_PAIRS = [12519, 10238, 10204]


def load_data():
    data = json.loads(DATA.read_text())
    if data["schema_version"] != 1 or data["background_sessions_per_output"] != 32782:
        raise ValueError("Unexpected evidence schema or background population")
    return data


def aggregate(row):
    folds = row["folds"]
    if len(folds) != 3 or {f["fold"] for f in folds} != {0, 1, 2}:
        raise ValueError("Every policy needs three distinct whole-device folds")
    for fold in folds:
        index = fold["fold"]
        if (
            fold["sessions"] != FOLD_SESSIONS[index]
            or fold["true_pairs"] != FOLD_PAIRS[index]
            or fold["population_sessions"] != FOLD_SESSIONS[index] + 32782
        ):
            raise ValueError("Mismatched scored population")
        if not 0 <= fold["tp"] <= fold["true_pairs"]:
            raise ValueError("Invalid pair count")
        if not 0 <= fold["sessions_wrongly_linked"] <= fold["sessions"]:
            raise ValueError("Invalid wrong-session count")
    return dict(
        id=row["id"],
        family=row["family"],
        tp=sum(f["tp"] for f in folds),
        true_pairs=sum(f["true_pairs"] for f in folds),
        wrong=sum(f["sessions_wrongly_linked"] for f in folds),
        sessions=sum(f["sessions"] for f in folds),
        max_fold_error=max(Fraction(f["sessions_wrongly_linked"], f["sessions"]) for f in folds),
    )


def recall(row):
    return Fraction(row["tp"], row["true_pairs"])


def error(row):
    return Fraction(row["wrong"], row["sessions"])


def best_at_cap(rows, cap):
    eligible = [r for r in rows if error(r) <= cap]
    if not eligible:
        raise ValueError("No measured policy meets this cap")
    return max(eligible, key=lambda r: (recall(r), -error(r), r["id"]))


def envelope(rows, cap):
    points, best = [], Fraction(-1)
    for row in sorted(rows, key=lambda r: (error(r), -recall(r))):
        if error(row) <= cap and recall(row) > best:
            points.append(row)
            best = recall(row)
    return points


def save(fig, name):
    IMG.mkdir(exist_ok=True)
    evidence = json.dumps(
        {
            "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    fig.savefig(IMG / f"{name}.png", metadata={"Description": evidence})
    fig.savefig(IMG / f"{name}.svg", metadata={"Description": evidence, "Date": None})
    plt.close(fig)


def write_cap_table(groups):
    with (ROOT / "docs/frontier_cap_winners.csv").open("w", newline="") as handle:
        fields = [
            "approach",
            "cap_percent",
            "recall_percent",
            "wrong_percent",
            "tp",
            "wrong_sessions",
            "policy",
            "worst_fold_wrong_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for family, rows in groups.items():
            for cap in CAPS:
                row = best_at_cap(rows, cap)
                writer.writerow(
                    dict(
                        approach=LABELS[family],
                        cap_percent=float(cap * 100),
                        recall_percent=float(recall(row) * 100),
                        wrong_percent=float(error(row) * 100),
                        tp=row["tp"],
                        wrong_sessions=row["wrong"],
                        policy=row["id"],
                        worst_fold_wrong_percent=float(row["max_fold_error"] * 100),
                    )
                )


def render_frontier(groups):
    fig = plt.figure(figsize=(12.8, 8.1))
    header(
        fig,
        "Device linking: the measured trade-off",
        "Refitted CatBoost leads near 1% error. Other models lead at higher limits.",
        "SAME DATA, SAME SCORING",
    )
    ax = fig.add_axes((0.075, 0.245, 0.665, 0.475))
    ax.set(xlim=(0, 5.12), ylim=(75, 94.2))
    ax.set_xlabel("Maximum wrongly linked sessions, %", labelpad=13)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    fig.text(0.075, 0.745, "Correct same-device pairs found (recall)", fontsize=10, color=MUTED)
    ax.axvline(1, color=GRID, lw=0.85, ls=(0, (3, 4)))
    endpoints = []
    for family, rows in groups.items():
        points = envelope(rows, CAPS[-1])
        x = [float(error(r) * 100) for r in points]
        y = [float(recall(r) * 100) for r in points]
        ax.step(
            x + [5],
            y + [y[-1]],
            where="post",
            color=COLORS[family],
            lw=2.5 if family == "main_refit" else 2,
            zorder=4 if family == "main_refit" else 3,
        )
        ax.scatter(x, y, color=COLORS[family], s=14, edgecolors=PAPER, linewidths=0.5, zorder=5)
        endpoints.append((y[-1], family))
    fig.text(0.79, 0.745, "AT A 5% CAP", fontsize=9, color=MUTED, weight="bold")
    for (value, family), label_y in zip(
        sorted(endpoints, reverse=True), [93, 89.5, 86, 82.5, 79], strict=True
    ):
        line_y = label_y + (1.35 if family == "main_refit" else 0.45)
        ax.plot([5.28, 5.43], [line_y] * 2, color=COLORS[family], lw=2, clip_on=False)
        ax.text(
            5.5,
            label_y + 0.28,
            LABELS[family].replace(" (refit)", "\n(refit)"),
            color=COLORS[family],
            fontsize=10.5,
            clip_on=False,
            weight="bold",
        )
        ax.text(5.5, label_y - 0.78, f"{value:.2f}% recall", fontsize=10, clip_on=False)
    fig.text(
        0.075,
        0.115,
        "Dots show tested settings; steps show the best recall measured under each error limit. No interpolation.\n"
        "Three tests hold out whole devices · 19,735 evaluated sessions · 32,961 true pairs · 32,782 background sessions per test.",
        fontsize=9,
        color=MUTED,
        linespacing=1.7,
    )
    fig.text(
        0.075,
        0.045,
        "Development data has been reused. Errors are pooled across three tests; any one test can exceed the limit.\n"
        "Red: submitted CatBoost approach, retrained for this comparison. Shipped weights are not scored here. Source: 19 October 2026 records.",
        fontsize=8.7,
        color=MUTED,
        linespacing=1.7,
    )
    save(fig, "frontier")


def validation_points(data):
    if len(data["validation"]) != 13 or len(data["current_validation"]) != 27:
        raise ValueError("Incomplete validation evidence")
    for row in data["validation"] + data["current_validation"]:
        if row["sessions"] != 6472 or row["true_pairs"] != 10027:
            raise ValueError("Validation population changed")
    return data["validation"], data["current_validation"]


def render_validation(data):
    rows, current = validation_points(data)
    fig = plt.figure(figsize=(12.8, 8.1))
    header(
        fig,
        "An error limit is a target, not a promise",
        "The 13 chosen settings stayed fixed for this check. Dots show measured results.",
        "SECOND CHECK · 13 SETTINGS FIXED",
    )
    left = fig.add_axes((0.075, 0.30, 0.53, 0.415))
    right = fig.add_axes((0.69, 0.30, 0.275, 0.415))
    markers = {
        "main_refit": "D",
        "nested_cost_gate": "^",
        "cost_size_utility": "s",
        "main_cost_hybrid": "*",
    }
    for ax in [left, right]:
        ax.scatter(
            [float(error(r) * 100) for r in current],
            [float(recall(r) * 100) for r in current],
            s=26,
            facecolors="none",
            edgecolors=COLORS["current_cost_guard"],
            lw=1,
        )
        for family, marker in markers.items():
            subset = [r for r in rows if r["family"] == family]
            ax.scatter(
                [float(error(r) * 100) for r in subset],
                [float(recall(r) * 100) for r in subset],
                s=120 if marker == "*" else 48,
                marker=marker,
                color=COLORS[family],
                edgecolors=PAPER,
                linewidths=0.6,
                zorder=4,
            )
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_xlabel("Wrongly linked sessions, %", labelpad=12)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.set(xlim=(-0.12, 11), ylim=(73, 97))
    left.xaxis.set_major_locator(MultipleLocator(2))
    left.yaxis.set_major_locator(MultipleLocator(5))
    right.set(xlim=(-0.04, 1.4), ylim=(80, 91))
    right.xaxis.set_major_locator(MultipleLocator(0.4))
    fig.text(0.075, 0.745, "Correct same-device pairs found (recall)", fontsize=10, color=MUTED)
    fig.text(0.69, 0.745, "LOW-ERROR DETAIL", fontsize=9, weight="bold", color=MUTED)
    hybrid = next(r for r in rows if r["family"] == "main_cost_hybrid")
    left.annotate(
        "Blend: aimed for 5%\nObserved: 5.41% error",
        xy=(float(error(hybrid) * 100), float(recall(hybrid) * 100)),
        xytext=(5.9, 85.2),
        fontsize=9,
        color=COLORS["main_cost_hybrid"],
        arrowprops=dict(arrowstyle="-", color=COLORS["main_cost_hybrid"], lw=0.8),
    )
    default = next(r for r in current if r["threshold"] == 0.2)
    left.annotate(
        "XGBoost + merge rules:\nbranch default",
        xy=(float(error(default) * 100), float(recall(default) * 100)),
        xytext=(2.4, 86.5),
        fontsize=9,
        color=COLORS["current_cost_guard"],
        arrowprops=dict(arrowstyle="-", color=COLORS["current_cost_guard"], lw=0.8),
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=COLORS["current_cost_guard"],
            label="XGBoost + merge rules (27 settings)",
        )
    ]
    handles.extend(
        plt.Line2D(
            [], [], marker=marker, linestyle="none", color=COLORS[family], label=LABELS[family]
        )
        for family, marker in markers.items()
    )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.17),
        ncol=3,
        fontsize=9.3,
        handletextpad=0.5,
        columnspacing=2,
    )
    fig.text(
        0.075,
        0.105,
        "The 13 filled points were chosen using the earlier three tests. Hollow points show a separate sweep of 27 XGBoost + merge rules settings.\n"
        "6,472 evaluated sessions · 10,027 true pairs · 32,782 background sessions. Two hollow points sit beyond the 11% scale.",
        fontsize=8.8,
        color=MUTED,
        linespacing=1.7,
    )
    fig.text(
        0.075,
        0.035,
        "Settings chosen for 3%, 5% and 10% limits exceeded them here. This data has been reused; it is not a fresh test.\n"
        "CatBoost uses training examples as references; all models are scored on the same sessions. Source: validation records.",
        fontsize=8.7,
        color=MUTED,
        linespacing=1.7,
    )
    save(fig, "frontier_validation")


def render_all():
    configure_style()
    data = load_data()
    rows = [aggregate(r) for r in data["policies"]]
    groups = {family: [r for r in rows if r["family"] == family] for family in LABELS}
    write_cap_table(groups)
    render_frontier(groups)
    render_validation(data)


if __name__ == "__main__":
    render_all()
