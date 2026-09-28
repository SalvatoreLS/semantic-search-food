from pathlib import Path

import pytest

from foodsearch.retrievers import BM25Retriever, Hit
from foodsearch.runs import hits_to_run, load_run, save_run, top_k_table
from foodsearch.systems import build_system, load_systems

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "systems.yaml"


def test_run_roundtrip(tmp_path: Path) -> None:
    run = hits_to_run({"q001": [Hit("a", 2.5, "bm25"), Hit("b", 1.0, "bm25")], "q002": []})
    save_run(run, tmp_path / "runs" / "bm25.json")
    assert load_run(tmp_path / "runs" / "bm25.json") == {"q001": {"a": 2.5, "b": 1.0}, "q002": {}}


def test_repo_config_builds_every_system() -> None:
    for name in load_systems(REPO_CONFIG):
        assert build_system(name, REPO_CONFIG).name == name


def test_bm25_params_come_from_config(tmp_path: Path) -> None:
    config = tmp_path / "systems.yaml"
    config.write_text("bm25_flat:\n  type: bm25\n  params:\n    b: 0.0\n")
    system = build_system("bm25_flat", config)
    assert isinstance(system, BM25Retriever) and system.b == 0.0


def test_unknown_system_or_type(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="Unknown system"):
        build_system("nope", REPO_CONFIG)
    config = tmp_path / "systems.yaml"
    config.write_text("x:\n  type: magic\n")
    with pytest.raises(ValueError, match="unknown type"):
        load_systems(config)


def test_top_k_table_ranks_by_score_then_item_id() -> None:
    run = {"q2": {"b": 1.0, "a": 1.0, "c": 3.0}, "q1": {"x": 0.5}}
    table = top_k_table(run, "sys", k=2)
    assert list(table.columns) == ["query_id", "rank", "itemId", "score", "system"]
    assert table[["query_id", "rank", "itemId"]].values.tolist() == [
        ["q1", 1, "x"],
        ["q2", 1, "c"],
        ["q2", 2, "a"],
    ]
