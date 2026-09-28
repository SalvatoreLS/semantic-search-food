import math

import numpy as np
import pandas as pd
import pytest

from foodsearch.eval.metrics import (
    compare,
    comparison_pairs,
    gap_counts,
    per_query,
    per_type,
    summarize,
)

QRELS = {"q1": {"a": 3, "b": 2, "c": 0}, "q2": {"d": 0}, "q3": {"f": 3}}
RUNS = {
    "s": {"q1": {"c": 3.0, "a": 2.0, "b": 1.0}, "q2": {"d": 1.0, "e": 0.5}, "q3": {"f": 1.0}},
    "t": {"q1": {"a": 3.0, "b": 2.0}, "q2": {}, "q3": {"g": 1.0}},
}
IS_FOOD = {"c": False}
TYPES = {"q1": "concrete_dish", "q2": "occasion", "q3": "product"}


@pytest.fixture
def table() -> pd.DataFrame:
    return per_query(RUNS, QRELS, IS_FOOD, TYPES).set_index(["system", "query_id"])


def test_hand_computed_metrics(table: pd.DataFrame) -> None:
    row = table.loc[("s", "q1")]
    ideal = 3 + 2 / math.log2(3)
    assert row["ndcg5"] == pytest.approx((3 / math.log2(3) + 2 / math.log2(4)) / ideal)
    assert row["p5"] == pytest.approx(0.4)
    assert row["mrr"] == pytest.approx(0.5)
    assert row["food_leak5"] == pytest.approx(0.2)
    assert row["judged10"] == 1.0
    assert table.loc[("t", "q1"), "ndcg5"] == pytest.approx(1.0)


def test_gap_and_product_rules(table: pd.DataFrame) -> None:
    gap = table.loc[("s", "q2")]
    assert np.isnan(gap["ndcg5"]) and np.isnan(gap["mrr"])
    assert gap["p5"] == 0.0 and gap["judged10"] == 0.5
    assert np.isnan(table.loc[("s", "q3"), "food_leak5"])
    assert np.isnan(table.loc[("t", "q2"), "judged10"])
    assert gap_counts(QRELS) == {"queries": 3, "no_graded_item": 1, "no_relevant_item": 1}


def test_summaries_and_comparisons() -> None:
    table = per_query(RUNS, QRELS, IS_FOOD, TYPES)
    summary = summarize(table).set_index(["system", "metric"])
    assert summary.loc[("s", "ndcg5"), "n"] == 2
    assert summary.loc[("s", "p5"), "n"] == 3
    tags = pd.DataFrame(
        {"query_id": ["q1", "q2", "q3"], "type": list(TYPES.values()), "us_translated": [1, 0, 0]}
    )
    groups = set(per_type(table, tags)["group"])
    assert groups == {"concrete_dish", "occasion", "product", "us_translated"}
    comparisons = compare(table, [("s", "t")])
    assert set(comparisons["metric"]) == {"ndcg5", "ndcg10", "p5", "mrr", "food_leak5"}
    assert (comparisons["p_holm"] >= comparisons["p"]).all()


def test_comparison_pairs_have_no_reversed_duplicates() -> None:
    pairs = comparison_pairs(
        ["bm25", "dense_pointwise", "main"], ["bm25", "dense_pointwise", "missing"]
    )
    assert pairs == [("dense_pointwise", "bm25"), ("main", "bm25"), ("main", "dense_pointwise")]


def test_per_query_rows_align_with_unsorted_qrels() -> None:
    qrels = {"q3": {"a": 3}, "q1": {"b": 3}, "q2": {"c": 3}}
    run = {"q2": {"c": 1.0}, "q1": {"x": 1.0}, "q3": {"z": 1.0}}
    table = per_query({"s": run}, qrels, {}, {}).set_index("query_id")
    assert table["ndcg5"].to_dict() == {"q1": 0.0, "q2": 1.0, "q3": 0.0}
