import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from foodsearch.llm import LLMClient, LLMResponseError, call_cost, cost_summary

MESSAGES = [{"role": "user", "content": "Responda em JSON"}]


class FakeChat:
    def __init__(self, contents: list[str], finish_reason: str = "stop") -> None:
        self.contents = contents
        self.finish_reason = finish_reason
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = self.contents[min(len(self.calls), len(self.contents)) - 1]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason, message=SimpleNamespace(content=content)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=100,
                prompt_tokens_details=SimpleNamespace(cached_tokens=200),
            ),
        )


class FakeEmbeddings:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def create(self, model: str, input: list[str], **kwargs: Any) -> SimpleNamespace:
        self.inputs.append(input)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0]) for i, t in enumerate(input)
        ]
        return SimpleNamespace(
            data=data[::-1], usage=SimpleNamespace(prompt_tokens=10 * len(input))
        )


def _client(tmp_path: Path, chat: FakeChat | None = None) -> tuple[LLMClient, SimpleNamespace]:
    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=chat or FakeChat(['{"ok": 1}'])),
        embeddings=FakeEmbeddings(),
    )
    return LLMClient(tmp_path / "cache", tmp_path / "cost.jsonl", client=fake), fake


def test_call_cost_uses_cached_input_price() -> None:
    assert call_cost("gpt-4.1-mini", 1000, 200, 100) == pytest.approx(
        (800 * 0.40 + 200 * 0.10 + 100 * 1.60) / 1e6
    )


def test_chat_is_cached_and_costed_once(tmp_path: Path) -> None:
    llm, fake = _client(tmp_path)
    first = llm.chat_json("gpt-4.1-mini", MESSAGES, tag="rerank")
    second = llm.chat_json("gpt-4.1-mini", MESSAGES, tag="rerank")
    assert first == second == {"ok": 1}
    assert len(fake.chat.completions.calls) == 1
    assert fake.chat.completions.calls[0]["temperature"] == 0
    assert len(llm.costs.calls) == 1
    logged = [json.loads(line) for line in (tmp_path / "cost.jsonl").read_text().splitlines()]
    assert logged[0]["tag"] == "rerank" and logged[0]["cached_input_tokens"] == 200


def test_cache_extra_changes_key(tmp_path: Path) -> None:
    llm, fake = _client(tmp_path)
    llm.chat_json("gpt-4.1", MESSAGES, tag="judge", cache_extra={"rubric_version": "v1"})
    llm.chat_json("gpt-4.1", MESSAGES, tag="judge", cache_extra={"rubric_version": "v2"})
    assert len(fake.chat.completions.calls) == 2


def test_cache_survives_new_client(tmp_path: Path) -> None:
    llm, _ = _client(tmp_path)
    llm.chat_json("gpt-4.1-mini", MESSAGES, tag="t")
    llm2, fake2 = _client(tmp_path)
    assert llm2.chat_json("gpt-4.1-mini", MESSAGES, tag="t") == {"ok": 1}
    assert fake2.chat.completions.calls == []


@pytest.mark.parametrize(
    ("contents", "finish_reason"),
    [(["not json"], "stop"), (["[1, 2]"], "stop"), (['{"ok": 1}'], "length")],
)
def test_failures_are_not_cached(tmp_path: Path, contents: list[str], finish_reason: str) -> None:
    llm, _ = _client(tmp_path, FakeChat(contents, finish_reason))
    with pytest.raises(LLMResponseError):
        llm.chat_json("gpt-4.1", MESSAGES, tag="judge")
    assert len(llm.cache) == 0
    assert len(llm.costs.calls) == 1


def test_api_exception_is_not_cached(tmp_path: Path) -> None:
    class Boom:
        def create(self, **kwargs: Any) -> None:
            raise TimeoutError("api down")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=Boom()))
    llm = LLMClient(tmp_path / "cache", tmp_path / "cost.jsonl", client=fake)
    with pytest.raises(TimeoutError):
        llm.chat_json("gpt-4.1", MESSAGES, tag="judge")
    assert len(llm.cache) == 0


def test_unknown_model_fails_before_calling(tmp_path: Path) -> None:
    llm, fake = _client(tmp_path)
    with pytest.raises(KeyError, match="No price"):
        llm.chat_json("gpt-unknown", MESSAGES, tag="t")
    assert fake.chat.completions.calls == []


def test_embed_sends_only_misses_and_keeps_order(tmp_path: Path) -> None:
    llm, fake = _client(tmp_path)
    first = llm.embed("text-embedding-3-small", ["aa", "b"], tag="emb")
    both = llm.embed("text-embedding-3-small", ["ccc", "aa", "ccc"], tag="emb")
    assert fake.embeddings.inputs == [["aa", "b"], ["ccc"]]
    np.testing.assert_array_equal(first[:, 0], [2.0, 1.0])
    np.testing.assert_array_equal(both[:, 0], [3.0, 2.0, 3.0])
    assert both.dtype == np.float32


def test_cost_summary_groups_by_tag_and_model(tmp_path: Path) -> None:
    llm, _ = _client(tmp_path)
    llm.chat_json("gpt-4.1-mini", MESSAGES, tag="query")
    llm.embed("text-embedding-3-small", ["x"], tag="emb")
    summary = cost_summary(tmp_path / "cost.jsonl").set_index("tag")
    assert set(summary.index) == {"query", "emb"}
    assert summary.loc["emb", "input_tokens"] == 10
    assert cost_summary(tmp_path / "missing.jsonl").empty


def test_validator_failure_is_not_cached(tmp_path: Path) -> None:
    llm, fake = _client(tmp_path, FakeChat(['{"grade": 7}', '{"grade": 2}']))

    def check(parsed: dict[str, Any]) -> None:
        if parsed["grade"] not in range(4):
            raise ValueError("grade out of range")

    with pytest.raises(ValueError):
        llm.chat_json("gpt-4.1", MESSAGES, tag="judge", validate=check)
    assert len(llm.cache) == 0
    assert llm.chat_json("gpt-4.1", MESSAGES, tag="judge", validate=check) == {"grade": 2}
    assert len(fake.chat.completions.calls) == 2
