import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from foodsearch.cards import item_card
from foodsearch.llm import UNDERSTANDING_MODEL, LLMClient, LLMResponseError, default_client
from foodsearch.query import QueryUnderstanding, rule_intent, understand
from foodsearch.rerank import PriorTrace, Reranker, apply_order, build_reranker, food_prior
from foodsearch.retrievers.base import Hit
from foodsearch.retrievers.bm25 import BM25Retriever
from foodsearch.retrievers.dense import DenseRetriever
from foodsearch.retrievers.fusion import SourceRanks, source_ranks, weighted_rrf

Understanding = Literal["none", "rule", "llm"]


@dataclass(frozen=True, slots=True)
class StageTrace:
    name: str
    model: str | None
    detail: str | None
    ms: float
    cost_usd: float
    llm_calls: int = 0
    failures: int = 0


@dataclass
class PipelineResult:
    hits: list[Hit]
    intent: str | None = None
    understanding: QueryUnderstanding | None = None
    prior: PriorTrace | None = None
    stages: list[StageTrace] = field(default_factory=list)
    sources: SourceRanks = field(default_factory=dict)


class _Stage:
    def __init__(self, llm: LLMClient | None) -> None:
        self._llm = llm

    def __enter__(self) -> "_Stage":
        self._start = time.perf_counter()
        self._calls = len(self._llm.costs.calls) if self._llm else 0
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
        self.cost = self._llm.cost_since(self._calls) if self._llm else 0.0

    def trace(
        self,
        name: str,
        model: str | None = None,
        detail: str | None = None,
        llm_calls: int = 0,
        failures: int = 0,
    ) -> StageTrace:
        return StageTrace(name, model, detail, round(self.ms, 1), self.cost, llm_calls, failures)


class Pipeline:
    def __init__(
        self,
        name: str,
        dense: Mapping[str, Any],
        bm25: Mapping[str, Any] | None = None,
        weights: Mapping[str, float] | None = None,
        understanding: Understanding = "none",
        expand: bool = False,
        prior_lambda: float | None = None,
        rerank: Mapping[str, Any] | None = None,
        candidates: int = 100,
        llm: LLMClient | None = None,
    ) -> None:
        if expand and understanding != "llm":
            raise ValueError("query expansion needs understanding: llm")
        self.name = name
        self._llm = llm
        self.dense = DenseRetriever(name=f"{name}.dense", llm=llm, **dense)
        self.bm25 = BM25Retriever(name=f"{name}.bm25", **bm25) if bm25 is not None else None
        self.weights = dict(weights or {})
        self.understanding = understanding
        self.expand = expand
        self.prior_lambda = prior_lambda
        self.reranker: Reranker | None = build_reranker(rerank, llm) if rerank else None
        self.candidates = candidates
        self._is_food: dict[str, bool] = {}
        self._cards: dict[str, str] = {}

    @property
    def llm(self) -> LLMClient:
        return self._llm or default_client()

    def fit(self, items: pd.DataFrame) -> None:
        self.dense.fit(items)
        if self.bm25 is not None:
            self.bm25.fit(items)
        self._is_food = dict(zip(items["item_id"], items["is_food"].astype(bool), strict=True))
        if self.reranker is not None:
            rows = items.set_index("item_id", drop=False)
            self._cards = {item_id: item_card(row) for item_id, row in rows.iterrows()}

    def search(self, query: str, k: int = 100) -> list[Hit]:
        return self.search_detailed(query, k).hits

    def _understand(self, query: str, result: PipelineResult) -> None:
        if self.understanding == "rule":
            with _Stage(None) as stage:
                result.intent = rule_intent(query)
            result.stages.append(stage.trace("intent rule", detail=f"intent {result.intent}"))
        elif self.understanding == "llm":
            failures = 0
            with _Stage(self.llm) as stage:
                try:
                    result.understanding = understand(query, self.llm)
                    result.intent = result.understanding.intent
                except LLMResponseError:
                    failures = 1
            detail = f"intent {result.intent}" if result.intent else "failed: raw query only"
            result.stages.append(
                stage.trace("query understanding", UNDERSTANDING_MODEL, detail, 1, failures)
            )

    def _retrieve(self, query: str, result: PipelineResult) -> dict[str, list[Hit]]:
        expanded = (
            result.understanding.expanded_text(query)
            if self.expand and result.understanding is not None
            else None
        )
        lists: dict[str, list[Hit]] = {}
        with _Stage(self.llm) as stage:
            texts = [query] if expanded is None else [query, expanded]
            vectors = self.dense.embed_queries(texts)
            if expanded is None:
                lists["dense"] = self.dense.search_vector(vectors[0], self.candidates)
            else:
                lists["dense raw"] = self.dense.search_vector(vectors[0], self.candidates)
                lists["dense expanded"] = self.dense.search_vector(vectors[1], self.candidates)
            if self.bm25 is not None:
                lists["bm25"] = self.bm25.search(expanded or query, self.candidates)
        detail = ", ".join(f"{label} {len(hits)}" for label, hits in lists.items())
        result.stages.append(stage.trace("retrieval", self.dense.model, detail))
        return lists

    def search_detailed(self, query: str, k: int = 100) -> PipelineResult:
        result = PipelineResult(hits=[])
        self._understand(query, result)
        lists = self._retrieve(query, result)

        if len(lists) > 1:
            with _Stage(None) as stage:
                fused = weighted_rrf(lists, self.weights, source=self.name)
            weights = ", ".join(f"{label} {self.weights.get(label, 1.0):g}" for label in lists)
            result.stages.append(stage.trace("fusion", detail=f"RRF k=60, {weights}"))
            hits, result.sources = fused.hits, fused.sources
        else:
            hits, result.sources = next(iter(lists.values())), source_ranks(lists)

        if self.prior_lambda is not None:
            with _Stage(None) as stage:
                hits, result.prior = food_prior(
                    hits, self._is_food, self.prior_lambda, result.intent
                )
            result.stages.append(stage.trace("food prior", detail=result.prior.reason))

        if self.reranker is not None:
            with _Stage(self.llm) as stage:
                reranked = self.reranker.rerank(query, hits, self._cards)
            model = self.reranker.model
            result.stages.append(
                stage.trace(
                    self.reranker.label,
                    model,
                    f"top {self.reranker.depth}",
                    reranked.llm_calls,
                    reranked.failures,
                )
            )
            for rank, item_id in enumerate(reranked.order, start=1):
                result.sources.setdefault(item_id, []).append((self.reranker.label, rank))
            hits = apply_order(hits, reranked.order, self.name)
        else:
            hits = [Hit(h.item_id, h.score, self.name) for h in hits]

        result.hits = hits[:k]
        return result
