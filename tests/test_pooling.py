from pathlib import Path

from foodsearch.eval.pooling import build_pool, load_pool, ranked, save_pool

RUNS = {
    "a": {"q1": {"x": 3.0, "y": 2.0, "z": 1.0}, "q2": {"m": 1.0}},
    "b": {"q1": {"z": 9.0, "w": 8.0, "x": 1.0}},
}


def test_ranked_breaks_ties_by_item_id() -> None:
    assert ranked({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]
    assert ranked({"b": 1.0, "a": 1.0, "c": 2.0}, depth=1) == ["c"]


def test_pool_keeps_best_rank_across_systems() -> None:
    pool = build_pool(RUNS, depth=2)
    assert pool.systems == ["a", "b"]
    assert pool.best_rank == {"q1": {"x": 1, "y": 2, "z": 1, "w": 2}, "q2": {"m": 1}}
    assert len(pool.pairs()) == 5


def test_pool_roundtrip(tmp_path: Path) -> None:
    pool = build_pool(RUNS)
    save_pool(pool, tmp_path / "pool.json")
    assert load_pool(tmp_path / "pool.json") == pool
