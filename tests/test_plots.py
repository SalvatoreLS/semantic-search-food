from pathlib import Path

import pandas as pd
import pytest

from foodsearch.eval.plots import frontier, grade_mix, grade_mix_plot, pareto_plot, spread


def test_frontier_treats_near_equal_costs_as_ties() -> None:
    cost = [0.0, 0.00002, 0.03, 0.14, 0.33]
    quality = [0.6, 0.8, 0.88, 0.93, 0.91]
    assert frontier(cost, quality) == [1, 2, 3]


def test_spread_keeps_a_minimum_gap() -> None:
    assert spread([0.5, 0.505, 0.9], gap=0.02) == pytest.approx([0.5, 0.52, 0.9])


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
