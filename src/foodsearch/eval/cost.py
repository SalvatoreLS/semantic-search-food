from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from foodsearch.llm import PRICES

Meta = Mapping[str, Any]
StageKey = tuple[str, str | None, str | None]


def _key(stage: Mapping[str, Any]) -> StageKey:
    return stage["name"], stage["model"], stage["detail"]


def _cached(stage: Mapping[str, Any]) -> bool:
    return stage["model"] in PRICES and stage["cost_usd"] == 0


def _per_query_usd(meta: Meta) -> float:
    return meta["usd"] / max(len(meta["queries"]), 1)


def dense_model(spec: Mapping[str, Any]) -> str | None:
    params = spec.get("params") or {}
    dense = params if spec["type"] == "dense" else params.get("dense") or {}
    return dense.get("model") if dense.get("backend", "openai") == "openai" else None


def cold_queries(
    metas: Mapping[str, Meta], systems: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    donors: dict[StageKey, dict[str, Mapping[str, Any]]] = {}
    for meta in metas.values():
        for qid, query in meta["queries"].items():
            for stage in query["stages"]:
                if stage["model"] in PRICES and stage["cost_usd"] > 0:
                    donors.setdefault(_key(stage), {}).setdefault(qid, stage)
    plain_dense = {
        dense_model(spec): name
        for name, spec in systems.items()
        if spec["type"] == "dense" and dense_model(spec) and name in metas
    }
    rows = []
    for system, meta in metas.items():
        for qid, query in meta["queries"].items():
            ms, usd, calls = query["ms"], 0.0, 0
            if not query["stages"]:
                usd = _per_query_usd(meta)
            for stage in query["stages"]:
                calls += stage["llm_calls"]
                if not _cached(stage):
                    usd += stage["cost_usd"]
                    continue
                donor = donors.get(_key(stage), {}).get(qid)
                fallback = plain_dense.get(stage["model"]) if stage["name"] == "retrieval" else None
                if donor is not None:
                    ms += donor["ms"] - stage["ms"]
                    usd += donor["cost_usd"]
                elif fallback is not None:
                    ms += metas[fallback]["queries"][qid]["ms"] - stage["ms"]
                    usd += _per_query_usd(metas[fallback])
            rows.append(
                {"system": system, "query_id": qid, "ms": ms, "usd": usd, "llm_calls": calls}
            )
    return pd.DataFrame(rows)


def index_usd(cost_log: pd.DataFrame) -> dict[str, float]:
    passages = cost_log[cost_log["tag"] == "embed:passage"]
    return passages.groupby("model")["usd"].sum().to_dict()


def cost_table(
    metas: Mapping[str, Meta],
    systems: Mapping[str, Mapping[str, Any]],
    cost_log: pd.DataFrame,
) -> pd.DataFrame:
    queries = cold_queries(metas, systems)
    index = index_usd(cost_log)
    rows = []
    for system, sub in queries.groupby("system", sort=False):
        model = dense_model(systems[system]) if system in systems else None
        rows.append(
            {
                "system": system,
                "llm_calls_per_query": sub["llm_calls"].mean(),
                "usd_per_100_queries": 100 * sub["usd"].mean(),
                "p50_ms": float(np.percentile(sub["ms"], 50)),
                "p95_ms": float(np.percentile(sub["ms"], 95)),
                "index_usd": index.get(model, 0.0) if model else 0.0,
                "device": metas[system]["device"],
            }
        )
    return pd.DataFrame(rows)
