from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from foodsearch.eval.judge import Qrels
from foodsearch.eval.pooling import ranked
from foodsearch.runs import Run

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
FRONTIER = "#2a78d6"
DOMINATED = "#9a9993"
GRADE_COLORS = ("#86b6ef", "#5598e7", "#256abf", "#104281")
GRADE_LABELS = ("0 irrelevant", "1 partial", "2 good", "3 perfect")
TOP = 5


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def frontier(cost: Sequence[float], quality: Sequence[float], tie: float = 0.01) -> list[int]:
    level = [round(c / tie) for c in cost]
    best, keep = float("-inf"), []
    for i in sorted(range(len(cost)), key=lambda j: (level[j], -quality[j])):
        if quality[i] > best:
            best = quality[i]
            keep.append(i)
    return keep


def spread(values: Sequence[float], gap: float) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in pairwise(order):
        placed[cur] = max(placed[cur], placed[prev] + gap)
    return placed


def pareto_plot(summary: pd.DataFrame, costs: pd.DataFrame, path: Path) -> None:
    ndcg = summary[summary["metric"] == "ndcg5"].set_index("system")
    data = costs.set_index("system").join(ndcg[["mean", "lo", "hi"]], how="inner").reset_index()
    x, y = data["usd_per_100_queries"].tolist(), data["mean"].tolist()
    on_front = set(frontier(x, y))
    fig = Figure(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    for i, row in data.iterrows():
        color = FRONTIER if i in on_front else DOMINATED
        ax.errorbar(
            row["usd_per_100_queries"],
            row["mean"],
            yerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            fmt="o",
            color=color,
            markersize=6,
            elinewidth=1,
            capsize=0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
    front = sorted(on_front, key=lambda i: x[i])
    ax.plot([x[i] for i in front], [y[i] for i in front], color=FRONTIER, linewidth=2, zorder=2)
    span = max(x) or 1.0
    label_y = spread(y, gap=0.018)
    for i, name in enumerate(data["system"]):
        ax.annotate(
            name,
            (x[i], y[i]),
            xytext=(x[i] + 0.02 * span, label_y[i]),
            textcoords="data",
            va="center",
            fontsize=8.5,
            color=TEXT if i in on_front else MUTED,
        )
    ax.set_xlim(-0.03 * span, 1.25 * span)
    ax.set_xlabel("LLM cost, $ per 100 queries (cold cache)", color=MUTED, fontsize=9)
    ax.set_ylabel("nDCG@5 (95% CI)", color=MUTED, fontsize=9)
    ax.set_title("Quality vs cost", loc="left", color=TEXT, fontsize=12, fontweight="semibold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def grade_mix(runs: Mapping[str, Run], qrels: Qrels, depth: int = TOP) -> pd.DataFrame:
    rows = []
    for system, run in runs.items():
        counts = [0] * len(GRADE_LABELS)
        for qid, grades in qrels.items():
            for iid in ranked(run.get(qid, {}), depth):
                if iid in grades:
                    counts[grades[iid]] += 1
        total = sum(counts) or 1
        rows.append({"system": system, **{str(g): c / total for g, c in enumerate(counts)}})
    return pd.DataFrame(rows)


def grade_mix_plot(mix: pd.DataFrame, order: Sequence[str], path: Path) -> None:
    data = mix.set_index("system").loc[[s for s in order if s in set(mix["system"])]]
    fig = Figure(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    ax.grid(axis="y", visible=False)
    left = pd.Series(0.0, index=data.index)
    for g, (label, color) in enumerate(zip(GRADE_LABELS, GRADE_COLORS, strict=True)):
        share = data[str(g)]
        ax.barh(
            data.index,
            share,
            left=left,
            color=color,
            height=0.62,
            edgecolor=SURFACE,
            linewidth=2,
            label=label,
        )
        left += share
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(f"share of top-{TOP} results by judge grade", color=MUTED, fontsize=9)
    ax.set_title(
        f"Grade mix of the top {TOP}", loc="left", color=TEXT, fontsize=12, fontweight="semibold"
    )
    ax.legend(
        ncols=4,
        loc="upper left",
        bbox_to_anchor=(0, -0.12),
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
