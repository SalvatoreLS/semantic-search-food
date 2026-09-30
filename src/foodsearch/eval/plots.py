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

SURFACE = "#FFFFFF"
TEXT = "#0B0F1A"
MUTED = "#5B6475"
GRID = "#E3E7EE"
ACCENT = "#0332AF"
OTHER = "#A3AAB8"
GRADE_COLORS = ("#B42318", "#B07A00", "#5A8F00", "#0F7B4F")
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


def spread(values: Sequence[float], gap: float) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for prev, cur in pairwise(order):
        placed[cur] = max(placed[cur], placed[prev] + gap)
    return placed


def pareto_plot(
    summary: pd.DataFrame,
    costs: pd.DataFrame,
    path: Path,
    systems: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
) -> None:
    ndcg = summary[summary["metric"] == "ndcg5"].set_index("system")
    data = costs.set_index("system").join(ndcg[["mean", "lo", "hi"]], how="inner")
    if systems is not None:
        data = data.loc[[s for s in systems if s in data.index]]
    data = data.reset_index()
    x, y = data["usd_per_100_queries"].tolist(), data["mean"].tolist()
    best = max(range(len(y)), key=lambda i: y[i])
    names = [(labels or {}).get(s, s) for s in data["system"]]
    fig = Figure(figsize=(8, 5), dpi=200, facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    for i, row in data.iterrows():
        ax.errorbar(
            row["usd_per_100_queries"],
            row["mean"],
            yerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            fmt="o",
            color=ACCENT if i == best else OTHER,
            markersize=9 if i == best else 6,
            elinewidth=1,
            capsize=0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4 if i == best else 3,
        )
    span = max(x) or 1.0
    label_y = spread(y, gap=0.018)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (x[i], y[i]),
            xytext=(x[i] + 0.02 * span, label_y[i]),
            textcoords="data",
            va="center",
            fontsize=9.5 if i == best else 8.5,
            fontweight="bold" if i == best else "normal",
            color=ACCENT if i == best else MUTED,
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


GAIN = ACCENT
LOSS = "#B42318"


def delta_plot(table: pd.DataFrame, system: str, base: str, path: Path) -> None:
    wide = table.pivot(index="query_id", columns="system", values="ndcg5")[[system, base]]
    delta = (wide[system] - wide[base]).dropna().sort_values(ascending=False)
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    fig = Figure(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    ax.grid(axis="x", visible=False)
    colors = [GAIN if d >= 0 else LOSS for d in delta]
    ax.bar(range(len(delta)), delta, color=colors, width=0.8, edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks([])
    ax.set_xlim(-1, len(delta))
    ax.set_xlabel(f"{len(delta)} queries, sorted by gain", color=MUTED, fontsize=9)
    ax.set_ylabel(f"nDCG@5 {system} minus {base}", color=MUTED, fontsize=9)
    ax.set_title(
        f"Per-query gain of {system} over {base}: {wins} better, {losses} worse, "
        f"{len(delta) - wins - losses} tied",
        loc="left",
        color=TEXT,
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
