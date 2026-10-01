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
    """What one pipeline stage did: its model, a short detail, time, cost, LLM calls and failures.
    The demo shows these, and `foodsearch run` stores them in the run meta, where the cost and
    latency report reads them.
    """

    name: str
    model: str | None
    detail: str | None
    ms: float
    cost_usd: float
    llm_calls: int = 0
    failures: int = 0


@dataclass
class PipelineResult:
    """The final hits plus everything needed to explain them: the intent, the dish expansion, the
    food prior decision, one trace per stage, and the rank each item had in every list before fusion
    and rerank.
    """

    hits: list[Hit]
    intent: str | None = None
    understanding: QueryUnderstanding | None = None
    prior: PriorTrace | None = None
    stages: list[StageTrace] = field(default_factory=list)
    sources: SourceRanks = field(default_factory=dict)


class _Stage:
    """Context manager around one stage: it measures the wall time and, when an LLM client is given,
    the dollars spent inside the block.
    """

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
        """Package the measured time and cost with the stage's name and details."""
        return StageTrace(name, model, detail, round(self.ms, 1), self.cost, llm_calls, failures)


class Pipeline:
    """The configurable search pipeline behind systems S4 to S9 in configs/systems.yaml.

    The headline system `hybrid` (S8) runs every stage: (1) query understanding, where gpt-4.1-mini
    returns the intent and a list of concrete Brazilian dishes; (2) dense retrieval on the raw and
    on the expanded query, plus BM25 on the expanded query; (3) weighted reciprocal rank fusion of
    those lists; (4) a soft food prior that pushes non-food items down for food queries; (5) an LLM
    listwise rerank of the top 30. Simpler systems switch stages off in their config, which is how
    the ablations are built.
    """

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
        """Build the stages from the params of one systems.yaml entry. Query expansion needs LLM
        understanding, because the expanded text is the LLM's dish list.
        """
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
        """The client given at construction, or the shared default one."""
        return self._llm or default_client()

    def fit(self, items: pd.DataFrame) -> None:
        """Index the catalog: document embeddings for dense retrieval, the BM25 index when this
        system uses it, the food flag of every item for the prior, and the item cards the reranker
        reads.
        """
        self.dense.fit(items)
        if self.bm25 is not None:
            self.bm25.fit(items)
        self._is_food = dict(zip(items["item_id"], items["is_food"].astype(bool), strict=True))
        if self.reranker is not None:
            rows = items.set_index("item_id", drop=False)
            self._cards = {item_id: item_card(row) for item_id, row in rows.iterrows()}

    def search(self, query: str, k: int = 100) -> list[Hit]:
        """The plain Retriever interface: the hits of search_detailed without the trace, so
        evaluation code can treat a pipeline like any other retriever.
        """
        return self.search_detailed(query, k).hits

    def _understand(self, query: str, result: PipelineResult) -> None:
        """Stage 1: decide the query intent. The rule variant only spots non-food product words; the
        LLM variant also returns the dish list used for expansion. If the LLM call fails, the search
        still runs on the raw query and the trace records the failure.
        """
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
        """Stage 2: build the candidate lists. Dense retrieval always runs on the raw query, and
        with expansion also on the query plus the LLM's dishes; keeping the raw list guards against
        an expansion that over-specifies. BM25, when configured, runs on the expanded text, where
        exact dish names give it something to match.
        """
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
        """Run the whole pipeline on one query and keep a trace of every stage.

        Understanding and retrieval come first. Then stage 3 merges the lists with weighted RRF,
        stage 4 rescales scores with the food prior, and stage 5 reorders the head with the
        reranker. The result carries the top `k` hits plus the per-stage trace and each item's
        rank in every list, which is what the demo's Search view displays.
        """
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
