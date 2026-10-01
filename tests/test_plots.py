from pathlib import Path

import pandas as pd
import pytest

from foodsearch.eval import plots
from foodsearch.eval.plots import (
    delta_plot,
    dodge,
    grade_mix,
    grade_mix_plot,
    pareto_plot,
    spread,
)


def test_spread_keeps_a_minimum_gap() -> None:
    assert spread([0.5, 0.505, 0.9], gap=0.02) == pytest.approx([0.5, 0.52, 0.9])


def test_dodge_moves_equal_costs_right_in_quality_order() -> None:
    placed = dodge([0.0, 0.0, 0.5, 0.0], [0.8, 0.6, 0.9, 0.7], width=0.01)
    assert placed == pytest.approx([0.02, 0.0, 0.5, 0.01])


def test_grade_mix_counts_judged_top_items() -> None:
    runs = {"s": {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}}
    mix = grade_mix(runs, {"q1": {"a": 3, "b": 0}}, depth=2).set_index("system")
    assert mix.loc["s", "3"] == 0.5 and mix.loc["s", "0"] == 0.5


def test_plots_render(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "system": ["a", "b"],
            "metric": "ndcg5",
            "mean": [0.6, 0.9],
            "lo": [0.5, 0.8],
            "hi": [0.7, 1.0],
        }
    )
    costs = pd.DataFrame({"system": ["a", "b"], "usd_per_100_queries": [0.0, 0.1]})
    pareto_plot(summary, costs, tmp_path / "pareto.png")
    mix = pd.DataFrame(
        {"system": ["a", "b"], "0": [0.5, 0], "1": [0.5, 0], "2": [0, 0.5], "3": [0, 0.5]}
    )
    grade_mix_plot(mix, ["b", "a"], tmp_path / "mix.png")
    assert (tmp_path / "pareto.png").stat().st_size > 0 and (tmp_path / "mix.png").exists()


def test_delta_plot_renders(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "system": ["a", "a", "b", "b"],
            "query_id": ["q1", "q2", "q1", "q2"],
            "ndcg5": [0.9, 0.4, 0.5, 0.6],
        }
    )
    delta_plot(table, "a", "b", tmp_path / "delta.png")
    assert (tmp_path / "delta.png").stat().st_size > 0


def test_pareto_plot_highlights_the_best_system_of_the_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = pd.DataFrame(
        {
            "system": ["a", "b", "c"],
            "metric": "ndcg5",
            "mean": [0.6, 0.8, 0.95],
            "lo": [0.5, 0.7, 0.9],
            "hi": [0.7, 0.9, 1.0],
        }
    )
    costs = pd.DataFrame({"system": ["a", "b", "c"], "usd_per_100_queries": [0.0, 0.1, 0.2]})
    drawn: list[tuple[str, str]] = []
    real = plots.Axes.annotate

    def spy(self, text, *args, **kwargs):  # type: ignore[no-untyped-def]
        drawn.append((text, kwargs["color"]))
        return real(self, text, *args, **kwargs)

    monkeypatch.setattr(plots.Axes, "annotate", spy)
    plots.pareto_plot(summary, costs, tmp_path / "p.png", ["a", "b"], {"b": "Bee"})
    assert drawn == [("a", plots.MUTED), ("Bee", plots.ACCENT)]
