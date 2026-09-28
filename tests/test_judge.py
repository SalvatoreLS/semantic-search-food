from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from foodsearch import paths
from foodsearch.data import load_queries
from foodsearch.eval.judge import (
    JudgePair,
    JudgeResult,
    Judgment,
    judge_pairs,
    load_judgments,
    save_judgments,
    write_qrels,
)
from foodsearch.eval.prompts import FEW_SHOTS, RUBRIC_VERSION, judge_messages, user_message
from foodsearch.llm import LLMClient

PAIRS = [
    JudgePair("q001", "i1", "Pizza de calabresa", "Nome: Pizza calabresa"),
    JudgePair("q001", "i2", "Pizza de calabresa", "Nome: Shampoo neutro"),
]


class FakeChat:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        content = self.contents[min(len(self.calls), len(self.contents)) - 1]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )


def _client(tmp_path: Path, contents: list[str]) -> tuple[LLMClient, FakeChat]:
    chat = FakeChat(contents)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=chat))
    return LLMClient(tmp_path / "cache", tmp_path / "cost.jsonl", client=fake), chat


def _no_sleep(_: float) -> None:
    return None


def test_judge_sees_only_query_and_card(tmp_path: Path) -> None:
    client, chat = _client(tmp_path, ['{"grade": 3, "reason": "ok"}'])
    result = judge_pairs(client, PAIRS[:1], "gpt-4.1", "en", workers=1, sleep=_no_sleep)
    assert result.grades == {"q001": {"i1": Judgment(grade=3, reason="ok")}}
    sent = chat.calls[0]["messages"]
    assert sent == judge_messages(PAIRS[0].query, PAIRS[0].card, "en")
    assert sent[-1]["content"] == user_message(PAIRS[0].query, PAIRS[0].card)
    assert len(sent) == 2 + 2 * len(FEW_SHOTS)


def test_prompt_language_and_rubric_are_part_of_the_cache_key(tmp_path: Path) -> None:
    client, chat = _client(tmp_path, ['{"grade": 2, "reason": "ok"}'])
    for lang in ("en", "pt", "en"):
        judge_pairs(client, PAIRS[:1], "gpt-4.1", lang, workers=1, sleep=_no_sleep)
    assert len(chat.calls) == 2
    assert RUBRIC_VERSION == "v1"


def test_invalid_grade_is_retried_then_left_unjudged(tmp_path: Path) -> None:
    client, chat = _client(tmp_path, ['{"grade": 5, "reason": "bad"}'])
    result = judge_pairs(client, PAIRS[:1], "gpt-4.1", "en", workers=1, sleep=_no_sleep)
    assert result.grades == {}
    assert [f["item_id"] for f in result.failures] == ["i1"]
    assert len(chat.calls) == 5
    assert len(client.cache) == 0


def test_retry_recovers_after_invalid_output(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, ["not json", '{"grade": 1, "reason": "partial"}'])
    result = judge_pairs(client, PAIRS[:1], "gpt-4.1", "en", workers=1, sleep=_no_sleep)
    assert result.grades["q001"]["i1"].grade == 1
    assert result.failures == []


def test_judgments_roundtrip_and_qrels(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, ['{"grade": 0, "reason": "no"}'])
    result = judge_pairs(client, PAIRS, "gpt-4.1", "en", workers=2, sleep=_no_sleep)
    save_judgments(result, tmp_path / "j.json", "all")
    loaded = load_judgments(tmp_path / "j.json")
    assert loaded.grades == result.grades
    assert write_qrels(loaded, tmp_path / "qrels.json") == {"q001": {"i1": 0, "i2": 0}}


def test_reranker_model_can_never_write_qrels(tmp_path: Path) -> None:
    result = JudgeResult(model="gpt-4.1-mini", lang="en")
    with pytest.raises(ValueError, match="reranker"):
        write_qrels(result, tmp_path / "qrels.json")


def test_few_shots_are_not_evaluation_queries_or_catalog_items() -> None:
    if not paths.queries_csv().exists() or not paths.items_parquet().exists():
        pytest.skip("confidential data not available")
    queries = set(load_queries(paths.queries_csv())["text"].str.lower())
    names = set(pd.read_parquet(paths.items_parquet(), columns=["name"])["name"].str.lower())
    for shot in FEW_SHOTS:
        assert shot.query.lower() not in queries
        assert str(shot.item["name"]).lower() not in names
