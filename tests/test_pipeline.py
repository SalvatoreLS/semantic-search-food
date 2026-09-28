import json
from typing import Any

import pandas as pd
import pytest

from foodsearch.pipeline import Pipeline

UNDERSTANDING = {
    "intent": "occasion",
    "dishes_pt": ["hambúrguer", "batata frita"],
    "keywords_pt": [],
    "meal_shift": "snack",
    "dietary": [],
}


def _handler(messages: list[dict[str, str]]) -> str:
    if "ranking" in messages[0]["content"]:
        return json.dumps({"ranking": [2, 1]})
    return json.dumps(UNDERSTANDING)


def _pipeline(llm: Any, **params: Any) -> Pipeline:
    dense = {"backend": "openai", "model": "text-embedding-3-small"}
    return Pipeline(name="main", dense=dense, llm=llm, **params)


def test_full_pipeline_traces_every_stage(make_llm: Any, items: pd.DataFrame) -> None:
    llm, fake = make_llm(_handler)
    pipeline = _pipeline(
        llm,
        understanding="llm",
        expand=True,
        prior_lambda=0.5,
        rerank={"type": "listwise", "depth": 3, "top": 2},
    )
    pipeline.fit(items)
    result = pipeline.search_detailed("Comida para piquenique", k=4)
    assert result.intent == "occasion"
    assert [s.name for s in result.stages] == [
        "query understanding",
        "retrieval",
        "fusion",
        "food prior",
        "listwise rerank",
    ]
    assert result.prior is not None and result.prior.on
    labels = {label for ranks in result.sources.values() for label, _ in ranks}
    assert labels == {"dense raw", "dense expanded", "listwise rerank"}
    assert {h.source for h in result.hits} == {"main"}
    assert result.hits[-1].item_id == "shampoo"
    assert result.stages[0].cost_usd > 0

    calls = len(fake.chat_calls)
    again = pipeline.search_detailed("Comida para piquenique", k=4)
    assert [h.item_id for h in again.hits] == [h.item_id for h in result.hits]
    assert len(fake.chat_calls) == calls
    assert all(s.cost_usd == 0 for s in again.stages)


def test_hybrid_bm25_list_has_no_zero_score_hits(make_llm: Any, items: pd.DataFrame) -> None:
    llm, _ = make_llm(_handler)
    pipeline = _pipeline(llm, bm25={}, understanding="llm", expand=True)
    pipeline.fit(items)
    result = pipeline.search_detailed("Comida para piquenique")
    bm25_items = {
        i for i, ranks in result.sources.items() if any(src == "bm25" for src, _ in ranks)
    }
    assert bm25_items and "pizza" not in bm25_items


def test_rule_intent_skips_prior_for_products(make_llm: Any, items: pd.DataFrame) -> None:
    llm, fake = make_llm(_handler)
    pipeline = _pipeline(llm, understanding="rule", prior_lambda=0.5)
    pipeline.fit(items)
    result = pipeline.search_detailed("Shampoo de morango")
    assert result.intent == "product" and result.prior is not None and not result.prior.on
    assert result.hits[0].item_id == "shampoo"
    assert not fake.chat_calls


def test_expansion_needs_llm_understanding() -> None:
    with pytest.raises(ValueError, match="expansion"):
        _pipeline(None, understanding="rule", expand=True)
