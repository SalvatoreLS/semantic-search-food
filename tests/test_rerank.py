import json
import re
from itertools import pairwise
from typing import Any

import pytest

from foodsearch.rerank import (
    ListwiseReranker,
    PointwiseReranker,
    apply_order,
    build_reranker,
    food_prior,
)
from foodsearch.retrievers import Hit

HITS = [Hit("a", 0.9, "s"), Hit("b", 0.8, "s"), Hit("c", 0.7, "s"), Hit("d", 0.6, "s")]
CARDS = {i: f"Nome: item {i}" for i in "abcd"}


def test_food_prior_penalizes_non_food_unless_product() -> None:
    is_food = {"a": False, "b": True, "c": True, "d": True}
    hits, trace = food_prior(HITS, is_food, 0.5, "dish")
    assert [h.item_id for h in hits] == ["b", "c", "d", "a"]
    assert hits[-1].score == pytest.approx(0.45) and trace.on
    same, trace = food_prior(HITS, is_food, 0.5, "product")
    assert same == HITS and not trace.on


def test_apply_order_puts_head_first_with_descending_scores() -> None:
    hits = apply_order(HITS, ["c", "a"], "sys")
    assert [h.item_id for h in hits] == ["c", "a", "b", "d"]
    assert all(x.score > y.score for x, y in pairwise(hits))
    assert {h.source for h in hits} == {"sys"}


def test_listwise_uses_ranking_then_fills(make_llm: Any) -> None:
    llm, _ = make_llm(lambda messages: json.dumps({"ranking": [3, 1]}))
    result = ListwiseReranker(depth=3, top=2, llm=llm).rerank("q", HITS, CARDS)
    assert result.order == ["c", "a", "b"] and result.failures == 0


def test_listwise_invalid_ranking_falls_back(make_llm: Any) -> None:
    llm, fake = make_llm(lambda messages: json.dumps({"ranking": [1, 9]}))
    result = ListwiseReranker(depth=3, llm=llm).rerank("q", HITS, CARDS)
    assert result.order == ["a", "b", "c"] and result.failures == 1
    assert len(fake.chat_calls) == 3


def test_pointwise_sorts_by_score_ties_by_retrieval_rank(make_llm: Any) -> None:
    scores = {"a": 3, "b": 8, "c": 8}

    def handler(messages: list[dict[str, str]]) -> str:
        item = re.search(r"item (\w)", messages[-1]["content"]).group(1)
        return json.dumps({"score": scores[item]} if item in scores else {"score": 99})

    llm, _ = make_llm(handler)
    result = PointwiseReranker(depth=4, workers=2, llm=llm).rerank("q", HITS, CARDS)
    assert result.order == ["b", "c", "a", "d"]
    assert result.llm_calls == 4 and result.failures == 1


def test_reranker_cannot_be_the_judge() -> None:
    with pytest.raises(ValueError, match="judge"):
        build_reranker({"type": "listwise", "model": "gpt-4.1"}, None)


def test_listwise_empty_ranking_keeps_prior_order(make_llm: Any) -> None:
    llm, fake = make_llm(lambda messages: json.dumps({"ranking": []}))
    result = ListwiseReranker(depth=3, llm=llm).rerank("q", HITS, CARDS)
    assert result.order == ["a", "b", "c"] and result.failures == 0
    assert len(fake.chat_calls) == 1
    assert "exactly 3" in fake.chat_calls[0][0]["content"]
