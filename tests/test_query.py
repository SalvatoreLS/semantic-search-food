import json
from typing import Any

import pytest

from foodsearch.llm import LLMResponseError
from foodsearch.query import QueryUnderstanding, rule_intent, understand

GOOD = {
    "intent": "occasion",
    "dishes_pt": ["hambúrguer", "cachorro-quente"],
    "keywords_pt": ["parque"],
    "meal_shift": "snack",
    "dietary": [],
}


def test_rule_intent_flags_non_food_products() -> None:
    assert rule_intent("Desinfetante para piso") == "product"
    assert rule_intent("Creme dental com flúor") == "product"
    assert rule_intent("Creme de milho") == "dish"
    assert rule_intent("Sanduíche de mortadela") == "dish"


def test_expanded_text_keeps_query_and_drops_duplicates() -> None:
    qu = QueryUnderstanding.model_validate({**GOOD, "dishes_pt": ["pizza", "pizza", "esfiha"]})
    assert qu.expanded_text("pizza") == "pizza; esfiha"


def test_invalid_answer_is_retried_and_never_cached(make_llm: Any) -> None:
    answers = iter([json.dumps({**GOOD, "intent": "vibes"}), json.dumps(GOOD)])
    llm, fake = make_llm(lambda messages: next(answers))
    result = understand("Comida para piquenique", llm)
    assert result.intent == "occasion"
    assert len(fake.chat_calls) == 2
    assert understand("Comida para piquenique", llm) == result
    assert len(fake.chat_calls) == 2


def test_persistent_failure_raises(make_llm: Any) -> None:
    llm, fake = make_llm(lambda messages: json.dumps({"intent": "dish"}))
    with pytest.raises(LLMResponseError):
        understand("Pizza", llm)
    assert len(fake.chat_calls) == 3
