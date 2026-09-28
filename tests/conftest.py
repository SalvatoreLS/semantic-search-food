from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from unidecode import unidecode

from foodsearch.llm import LLMClient

VOCAB = ("hamburguer", "pizza", "shampoo", "batata", "estadio")

ChatHandler = Callable[[list[dict[str, str]]], str]


def keyword_vector(text: str) -> list[float]:
    folded = unidecode(text).lower()
    return [float(folded.count(word)) for word in VOCAB] + [0.01]


class FakeOpenAI:
    def __init__(self, handler: ChatHandler) -> None:
        self.handler = handler
        self.chat_calls: list[list[dict[str, str]]] = []
        self.embed_calls: list[list[str]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _chat(self, messages: list[dict[str, str]], **kwargs: Any) -> SimpleNamespace:
        self.chat_calls.append(messages)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=self.handler(messages))
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=500, completion_tokens=20, prompt_tokens_details=None
            ),
        )

    def _embed(self, model: str, input: Sequence[str], **kwargs: Any) -> SimpleNamespace:
        self.embed_calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=keyword_vector(t)) for i, t in enumerate(input)]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=5 * len(input)))


@pytest.fixture
def make_llm(tmp_path: Path) -> Callable[[ChatHandler], tuple[LLMClient, FakeOpenAI]]:
    def make(handler: ChatHandler) -> tuple[LLMClient, FakeOpenAI]:
        fake = FakeOpenAI(handler)
        return LLMClient(tmp_path / "cache", tmp_path / "cost.jsonl", client=fake), fake

    return make


def _item(item_id: str, name: str, is_food: bool) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "name": name,
        "doc": f"{name}.",
        "category_name": "Lanches" if is_food else "Higiene",
        "l1": None,
        "l2": None,
        "description_clean": "",
        "price": 20.0,
        "price_bucket": "médio",
        "vegan": False,
        "lac_free": False,
        "organic": False,
        "tags": {},
        "is_food": is_food,
    }


@pytest.fixture
def items() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _item("burger", "Hambúrguer artesanal", True),
            _item("pizza", "Pizza de calabresa", True),
            _item("fries", "Batata frita", True),
            _item("shampoo", "Shampoo hamburguer de morango", False),
        ]
    )


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("foodsearch.llm.time.sleep", lambda seconds: None)
