from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from foodsearch.retrievers.base import Hit

RRF_K = 60

SourceRanks = dict[str, list[tuple[str, int]]]


@dataclass(frozen=True, slots=True)
class Fused:
    hits: list[Hit]
    sources: SourceRanks


def source_ranks(lists: Mapping[str, Sequence[Hit]]) -> SourceRanks:
    sources: SourceRanks = defaultdict(list)
    for label, hits in lists.items():
        for rank, hit in enumerate(hits, start=1):
            sources[hit.item_id].append((label, rank))
    return dict(sources)


def weighted_rrf(
    lists: Mapping[str, Sequence[Hit]],
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
    source: str = "rrf",
) -> Fused:
    weights = weights or {}
    scores: dict[str, float] = defaultdict(float)
    for label, hits in lists.items():
        weight = weights.get(label, 1.0)
        for rank, hit in enumerate(hits, start=1):
            scores[hit.item_id] += weight / (k + rank)
    order = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    return Fused(
        hits=[Hit(item_id=i, score=scores[i], source=source) for i in order],
        sources=source_ranks(lists),
    )
