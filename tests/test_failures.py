import pandas as pd

from foodsearch.eval.failures import failure_table


def test_worst_queries_with_pool_diagnostics() -> None:
    table = pd.DataFrame(
        {
            "system": ["s", "s", "t", "t"],
            "query_id": ["q1", "q2", "q1", "q2"],
            "ndcg5": [0.3, 0.9, 0.8, 0.5],
            "p5": [0.2, 0.8, 0.6, 0.4],
            "food_leak5": [0.2, 0.0, 0.0, 0.0],
        }
    )
    qrels = {"q1": {"a": 0, "b": 3, "c": 2}, "q2": {"d": 3}}
    runs = {"s": {"q1": {"a": 2.0, "b": 1.0}, "q2": {"d": 1.0}}}
    tags = pd.DataFrame(
        {"query_id": ["q1", "q2"], "type": ["occasion", "concrete_dish"], "us_translated": [1, 0]}
    )
    result = failure_table(table, qrels, runs, tags, "s", n=1)
    row = result.iloc[0]
    assert list(result["query_id"]) == ["q1"]
    assert (row["pool_relevant"], row["pool_perfect"], row["top5_relevant"]) == (2, 1, 1)
    assert row["catalog_gap"] and row["best_system"] == "t" and row["best_ndcg5"] == 0.8
