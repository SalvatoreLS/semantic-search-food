from pathlib import Path

import pandas as pd
import pytest

from foodsearch.eval.human import (
    LabelPair,
    build_label_queue,
    export_sheet,
    human_qrels,
    import_sheet,
    intra_pairs,
    load_labels,
    load_queue,
    record_label,
    sample_pairs,
    save_queue,
)
from foodsearch.eval.pooling import Pool


def _pool() -> Pool:
    big = {f"h{i}": 1 + i % 3 for i in range(12)} | {f"t{i}": 4 + i % 7 for i in range(12)}
    few_head = {"a": 1, "b": 2} | {f"t{i}": 5 for i in range(20)}
    small = {"x": 1, "y": 7}
    return Pool(depth=10, systems=["s"], best_rank={"q1": big, "q2": few_head, "q3": small})


def test_sampling_splits_head_and_tail_and_tops_up() -> None:
    pairs = sample_pairs(_pool(), ["q1", "q2", "q3"], seed=1)
    by_query: dict[str, list[LabelPair]] = {}
    for p in pairs:
        by_query.setdefault(p.query_id, []).append(p)
    assert sum(p.best_rank <= 3 for p in by_query["q1"]) == 8
    assert sum(p.best_rank > 3 for p in by_query["q1"]) == 7
    assert len(by_query["q2"]) == 15 and sum(p.best_rank <= 3 for p in by_query["q2"]) == 2
    assert {p.item_id for p in by_query["q3"]} == {"x", "y"}
    assert pairs == sample_pairs(_pool(), ["q1", "q2", "q3"], seed=1)


def test_queue_appends_repeats_as_second_pass(tmp_path: Path) -> None:
    pairs = sample_pairs(_pool(), ["q1", "q2", "q3"])
    queue = build_label_queue(pairs, repeats=5)
    first, second = queue[: len(pairs)], queue[len(pairs) :]
    assert sorted(p.pair_id for p in first) == sorted(p.pair_id for p in pairs)
    assert len(second) == 5 and all(p.pass_ == 2 for p in second)
    assert {(p.query_id, p.item_id) for p in second} <= {(p.query_id, p.item_id) for p in first}
    save_queue(queue, tmp_path / "queue.csv")
    assert load_queue(tmp_path / "queue.csv") == queue
    with pytest.raises(FileExistsError):
        save_queue(queue, tmp_path / "queue.csv")


def test_record_label_overwrites_same_pair_and_keeps_passes(tmp_path: Path) -> None:
    path = tmp_path / "human.csv"
    record_label(path, "q1:i:1:1", 1, "ann")
    record_label(path, "q1:i:1:1", 3, "ann")
    record_label(path, "q1:i:1:2", 2, "ann")
    labels = load_labels(path)
    assert len(labels) == 2
    assert human_qrels(labels) == {"q1": {"i:1": 3}}
    intra = intra_pairs(labels)
    assert intra[["grade_1", "grade_2"]].values.tolist() == [[3, 2]]
    with pytest.raises(ValueError):
        record_label(path, "q1:i:1:1", 4, "ann")
    with pytest.raises(ValueError):
        record_label(path, "q1:i:3", 1, "ann")


def test_sheet_roundtrip(tmp_path: Path) -> None:
    queue = [LabelPair("q001", "i1", 1, 2)]
    queries = pd.DataFrame({"query_id": ["q001"], "text": ["Pizza"]})
    items = pd.DataFrame(
        {
            "item_id": ["i1"],
            "name": ["Pizza"],
            "category_name": ["Pizzas"],
            "l1": [None],
            "l2": [None],
            "description_clean": [""],
            "price": [10.0],
            "price_bucket": ["barato"],
            "vegan": [False],
            "lac_free": [False],
            "organic": [False],
            "tags": [{}],
            "is_food": [True],
        }
    )
    sheet = tmp_path / "sheet.csv"
    export_sheet(queue, queries, items, sheet)
    exported = pd.read_csv(sheet)
    assert "best_rank" not in exported.columns and "score" not in exported.columns
    exported["grade"] = [2]
    exported.to_csv(sheet, index=False)
    assert import_sheet(sheet, tmp_path / "human.csv", "ann") == 1
    assert human_qrels(load_labels(tmp_path / "human.csv")) == {"q001": {"i1": 2}}
