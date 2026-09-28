import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from foodsearch.runs import Run

POOL_DEPTH = 10


@dataclass(frozen=True, slots=True)
class Pool:
    depth: int
    systems: list[str]
    best_rank: dict[str, dict[str, int]]

    def pairs(self) -> list[tuple[str, str]]:
        return [(qid, iid) for qid, items in self.best_rank.items() for iid in items]


def ranked(scores: Mapping[str, float], depth: int | None = None) -> list[str]:
    order = sorted(scores, key=lambda iid: (-scores[iid], iid))
    return order if depth is None else order[:depth]


def build_pool(runs: Mapping[str, Run], depth: int = POOL_DEPTH) -> Pool:
    best: dict[str, dict[str, int]] = {}
    for run in runs.values():
        for qid, scores in run.items():
            items = best.setdefault(qid, {})
            for rank, iid in enumerate(ranked(scores, depth), start=1):
                items[iid] = min(rank, items.get(iid, rank))
    return Pool(depth=depth, systems=sorted(runs), best_rank=dict(sorted(best.items())))


def save_pool(pool: Pool, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"depth": pool.depth, "systems": pool.systems, "pool": pool.best_rank}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_pool(path: Path) -> Pool:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Pool(depth=raw["depth"], systems=raw["systems"], best_rank=raw["pool"])
