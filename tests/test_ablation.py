import numpy as np
import pandas as pd
import pytest

from foodsearch.eval.ablation import Step, ablations, rerank_gain, restrict

STEPS = (
    Step("component", "b", "a", "+ b"),
    Step("component", "c", "b", "+ c"),
    Step("other", "c", "a", "c vs a"),
    Step("component", "missing", "a", "skipped"),
)


def make_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.6, 40)
    frames = [
        pd.DataFrame({"system": name, "query_id": [f"q{i}" for i in range(40)], "ndcg5": values})
        for name, values in (("a", base), ("b", base + 0.2), ("c", base + 0.2))
    ]
    return pd.concat(frames, ignore_index=True)


def test_steps_are_paired_and_corrected_per_family() -> None:
    result = ablations(make_table(), STEPS, metrics=["ndcg5"]).set_index("label")
    assert list(result.index) == ["+ b", "+ c", "c vs a"]
    assert result.loc["+ b", "diff"] == pytest.approx(0.2)
    assert result.loc["+ b", "ci_excludes_0"] and result.loc["+ b", "significant"]
    assert result.loc["+ c", "diff"] == pytest.approx(0.0)
    assert not result.loc["+ c", "ci_excludes_0"]
    assert result.loc["c vs a", "p_holm"] == pytest.approx(result.loc["c vs a", "p"])
    assert (result["p_holm"] >= result["p"]).all()


def test_restrict_keeps_only_labelled_pairs() -> None:
    judge = {"q1": {"a": 3, "b": 1}, "q2": {"c": 2}}
    assert restrict(judge, {"q1": {"a": 0, "z": 2}}) == {"q1": {"a": 3}}


def test_rerank_gain_pairs_both_qrels() -> None:
    steps = (Step("rerank", "b", "a", "+ b"),)
    table = make_table()
    human = table.assign(ndcg5=table["ndcg5"] * 0.5)
    gain = rerank_gain(table, human, steps).set_index("label")
    assert gain.loc["+ b", "diff_judge"] == pytest.approx(0.2)
    assert gain.loc["+ b", "diff_human"] == pytest.approx(0.1)
