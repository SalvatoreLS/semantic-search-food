import pandas as pd
import pytest

from foodsearch.eval.cost import cold_queries, cost_table

LARGE = "text-embedding-3-large"
MINI = "gpt-4.1-mini"


def stage(name: str, model: str | None, ms: float, usd: float, calls: int = 0) -> dict:
    detail = "dense 100" if name == "retrieval" else None
    return {
        "name": name,
        "model": model,
        "detail": detail,
        "ms": ms,
        "cost_usd": usd,
        "llm_calls": calls,
    }


def meta(queries: dict, usd: float = 0.0) -> dict:
    return {"device": "cpu", "usd": usd, "queries": queries}


SYSTEMS = {
    "dense": {"type": "dense", "params": {"backend": "openai", "model": LARGE}},
    "local": {"type": "dense", "params": {"backend": "st", "model": "e5"}},
    "cold": {"type": "pipeline", "params": {"dense": {"model": LARGE}}},
    "warm": {"type": "pipeline", "params": {"dense": {"model": LARGE}}},
}
METAS = {
    "dense": meta({"q1": {"ms": 200.0, "stages": []}}, usd=0.002),
    "local": meta({"q1": {"ms": 50.0, "stages": []}}),
    "cold": meta(
        {
            "q1": {
                "ms": 1210.0,
                "stages": [
                    stage("retrieval", LARGE, 10.0, 0.0),
                    stage("understanding", MINI, 1200.0, 0.01, 1),
                ],
            }
        }
    ),
    "warm": meta(
        {
            "q1": {
                "ms": 511.0,
                "stages": [
                    stage("retrieval", LARGE, 10.0, 0.0),
                    stage("understanding", MINI, 1.0, 0.0, 1),
                    stage("rerank", MINI, 500.0, 0.02, 1),
                ],
            }
        }
    ),
}


def test_cached_stages_take_the_cold_values() -> None:
    rows = cold_queries(METAS, SYSTEMS).set_index("system")
    assert rows.loc["warm", "ms"] == pytest.approx(511.0 - 1.0 + 1200.0 - 10.0 + 200.0)
    assert rows.loc["warm", "usd"] == pytest.approx(0.02 + 0.01 + 0.002)
    assert rows.loc["warm", "llm_calls"] == 2
    assert rows.loc["cold", "usd"] == pytest.approx(0.01 + 0.002)
    assert rows.loc["local", "usd"] == 0.0


def test_cost_table_adds_index_cost_per_backend() -> None:
    log = pd.DataFrame(
        {"tag": ["embed:passage", "embed:query"], "model": [LARGE, LARGE], "usd": [0.03, 0.001]}
    )
    table = cost_table(METAS, SYSTEMS, log).set_index("system")
    assert table.loc["dense", "index_usd"] == pytest.approx(0.03)
    assert table.loc["local", "index_usd"] == 0.0
    assert table.loc["warm", "usd_per_100_queries"] == pytest.approx(3.2)
