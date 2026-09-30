from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from foodsearch.llm import LLMClient
from foodsearch.pipeline import Pipeline
from foodsearch.retrievers import BM25Retriever, DenseRetriever, Retriever

RetrieverFactory = Callable[..., Retriever]

HEADLINE_SYSTEM = "hybrid"
COMPARE_SYSTEM = "dense_pointwise"

REGISTRY: dict[str, RetrieverFactory] = {
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "pipeline": Pipeline,
}


def load_systems(path: Path) -> dict[str, dict[str, Any]]:
    systems = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for name, spec in systems.items():
        if spec.get("type") not in REGISTRY:
            raise ValueError(f"System {name!r} has unknown type {spec.get('type')!r}")
    return systems


def build_system(name: str, config_path: Path) -> Retriever:
    systems = load_systems(config_path)
    if name not in systems:
        raise KeyError(f"Unknown system {name!r}; defined: {sorted(systems)}")
    return build_from_spec(name, systems[name])


def build_from_spec(name: str, spec: Mapping[str, Any], llm: LLMClient | None = None) -> Retriever:
    params = dict(spec.get("params") or {})
    if spec["type"] != "bm25" and llm is not None:
        params["llm"] = llm
    return REGISTRY[spec["type"]](name=name, **params)
