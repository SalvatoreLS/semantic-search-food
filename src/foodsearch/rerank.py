from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from foodsearch.device import resolve_device
from foodsearch.eval.judge import JUDGE_MODEL, RERANKER_MODEL
from foodsearch.llm import LLMClient, LLMResponseError, default_client
from foodsearch.retrievers.base import Hit


@dataclass(frozen=True, slots=True)
class PriorTrace:
    on: bool
    lam: float
    reason: str


def food_prior(
    hits: Sequence[Hit], is_food: Mapping[str, bool], lam: float, intent: str | None
) -> tuple[list[Hit], PriorTrace]:
    if intent == "product":
        return list(hits), PriorTrace(False, lam, "product intent: no penalty")
    scored = [
        Hit(h.item_id, h.score if is_food.get(h.item_id, True) else h.score * lam, h.source)
        for h in hits
    ]
    scored.sort(key=lambda h: (-h.score, h.item_id))
    reason = f"{intent} intent" if intent else "food intent assumed"
    return scored, PriorTrace(True, lam, f"{reason}: non-food scores x {lam}")


def rank_scores(order: Sequence[str], source: str) -> list[Hit]:
    n = len(order)
    return [Hit(item_id, (n - i) / n, source) for i, item_id in enumerate(order)]


def apply_order(hits: Sequence[Hit], head: Sequence[str], source: str) -> list[Hit]:
    placed = set(head)
    return rank_scores([*head, *(h.item_id for h in hits if h.item_id not in placed)], source)


@dataclass(frozen=True, slots=True)
class RerankResult:
    order: list[str]
    llm_calls: int
    failures: int


class Reranker(Protocol):
    label: str
    model: str
    depth: int

    def rerank(self, query: str, hits: Sequence[Hit], cards: Mapping[str, str]) -> RerankResult: ...


LISTWISE_PROMPT = """\
You rank products for a Brazilian food-delivery search engine. The catalog mixes restaurant \
dishes with groceries, pharmacy, hygiene, pet and home items. Given a Portuguese query and \
numbered candidate items, pick the {top} items a customer who typed this query would most \
want to see, best first. Prefer items that directly satisfy the query's dish, cuisine or \
occasion; prefer prepared dishes over ingredients when the query asks for a dish; never \
prefer non-food items for a food query. If no candidate matches well, still rank the {top} \
closest ones. Return JSON: {{"ranking": [candidate numbers]}} with exactly {top} distinct \
numbers taken only from the list."""


class ListwiseRanking(BaseModel):
    ranking: list[int]


class ListwiseReranker:
    label = "listwise rerank"

    def __init__(
        self,
        model: str = RERANKER_MODEL,
        depth: int = 30,
        top: int = 10,
        llm: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.depth = depth
        self.top = top
        self._llm = llm

    def messages(self, query: str, candidates: Sequence[str]) -> list[dict[str, str]]:
        listing = "\n\n".join(f"[{i}]\n{card}" for i, card in enumerate(candidates, start=1))
        return [
            {
                "role": "system",
                "content": LISTWISE_PROMPT.format(top=min(self.top, len(candidates))),
            },
            {"role": "user", "content": f"Query: {query}\n\nCandidates:\n\n{listing}"},
        ]

    def _parser(self, n: int, top: int) -> Any:
        def parse(payload: dict[str, Any]) -> list[int]:
            ranking = ListwiseRanking.model_validate(payload).ranking
            if len(set(ranking)) != len(ranking) or not all(1 <= i <= n for i in ranking):
                raise ValueError(f"ranking must hold distinct numbers in 1..{n}: {ranking}")
            return ranking[:top]

        return parse

    def rerank(self, query: str, hits: Sequence[Hit], cards: Mapping[str, str]) -> RerankResult:
        head = [h.item_id for h in hits[: self.depth]]
        if not head:
            return RerankResult([], 0, 0)
        top = min(self.top, len(head))
        try:
            ranking = (self._llm or default_client()).chat_checked(
                self.model,
                self.messages(query, [cards[i] for i in head]),
                self._parser(len(head), top),
                tag="rerank:listwise",
                max_tokens=120,
            )
        except LLMResponseError:
            return RerankResult(head, 1, 1)
        chosen = [head[i - 1] for i in ranking]
        placed = set(chosen)
        return RerankResult([*chosen, *(i for i in head if i not in placed)], 1, 0)


POINTWISE_PROMPT = """\
Rate how well one product matches a Portuguese search query in a Brazilian food-delivery \
app, from 0 (unrelated) to 10 (exactly what the customer wants). Return JSON: {"score": n}."""


class PointwiseScore(BaseModel):
    score: int = Field(ge=0, le=10)


class PointwiseReranker:
    label = "pointwise rerank"

    def __init__(
        self,
        model: str = RERANKER_MODEL,
        depth: int = 50,
        workers: int = 8,
        llm: LLMClient | None = None,
    ) -> None:
        self.model = model
        self.depth = depth
        self.workers = workers
        self._llm = llm

    def messages(self, query: str, card: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": POINTWISE_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nProduct:\n{card}"},
        ]

    def _score(self, query: str, card: str) -> int | None:
        try:
            return (self._llm or default_client()).chat_checked(
                self.model,
                self.messages(query, card),
                lambda payload: PointwiseScore.model_validate(payload).score,
                tag="rerank:pointwise",
                max_tokens=20,
            )
        except LLMResponseError:
            return None

    def rerank(self, query: str, hits: Sequence[Hit], cards: Mapping[str, str]) -> RerankResult:
        head = [h.item_id for h in hits[: self.depth]]
        with ThreadPoolExecutor(self.workers) as pool:
            scores = list(pool.map(lambda i: self._score(query, cards[i]), head))
        ranked = [-1 if s is None else s for s in scores]
        order = sorted(range(len(head)), key=lambda j: (-ranked[j], j))
        return RerankResult([head[j] for j in order], len(head), sum(s is None for s in scores))


class CrossEncoderReranker:
    label = "cross-encoder rerank"

    def __init__(
        self, model: str = "BAAI/bge-reranker-v2-m3", depth: int = 30, batch_size: int = 16
    ) -> None:
        self.model = model
        self.depth = depth
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            device = resolve_device()
            self._model = CrossEncoder(self.model, device=device)
            if device == "cuda":
                self._model.model.half()
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit], cards: Mapping[str, str]) -> RerankResult:
        head = [h.item_id for h in hits[: self.depth]]
        if not head:
            return RerankResult([], 0, 0)
        scores = self._load().predict(
            [(query, cards[i]) for i in head], batch_size=self.batch_size, show_progress_bar=False
        )
        order = sorted(range(len(head)), key=lambda j: (-float(scores[j]), j))
        return RerankResult([head[j] for j in order], 0, 0)


RERANKERS: dict[str, type] = {
    "listwise": ListwiseReranker,
    "pointwise": PointwiseReranker,
    "cross_encoder": CrossEncoderReranker,
}


def build_reranker(spec: Mapping[str, Any], llm: LLMClient | None) -> Reranker:
    params = dict(spec)
    kind = params.pop("type")
    if kind not in RERANKERS:
        raise ValueError(f"Unknown reranker type {kind!r}; expected one of {sorted(RERANKERS)}")
    if kind != "cross_encoder":
        if params.get("model", RERANKER_MODEL) == JUDGE_MODEL:
            raise ValueError(f"the reranker must not be the judge model {JUDGE_MODEL}")
        params["llm"] = llm
    return RERANKERS[kind](**params)
