from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from foodsearch.retrievers import BM25Retriever, Retriever

RetrieverFactory = Callable[..., Retriever]

REGISTRY: dict[str, RetrieverFactory] = {
    "bm25": BM25Retriever,
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
    spec = systems[name]
    return REGISTRY[spec["type"]](name=name, **(spec.get("params") or {}))
