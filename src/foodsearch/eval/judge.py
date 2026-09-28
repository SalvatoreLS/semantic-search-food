import json
import random
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import openai
import pandas as pd
from pydantic import BaseModel, ValidationError

from foodsearch.cards import item_card
from foodsearch.eval.prompts import RUBRIC_VERSION, Lang, judge_messages
from foodsearch.llm import LLMClient, LLMResponseError

JUDGE_MODEL = "gpt-4.1"
RERANKER_MODEL = "gpt-4.1-mini"
MAX_ATTEMPTS = 5
SEED = 20260926

Qrels = dict[str, dict[str, int]]


class Judgment(BaseModel):
    grade: Literal[0, 1, 2, 3]
    reason: str


@dataclass(frozen=True, slots=True)
class JudgePair:
    query_id: str
    item_id: str
    query: str
    card: str


@dataclass
class JudgeResult:
    model: str
    lang: Lang
    grades: dict[str, dict[str, Judgment]] = field(default_factory=dict)
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def n_judged(self) -> int:
        return sum(len(items) for items in self.grades.values())


def make_pairs(
    pairs: Iterable[tuple[str, str]], queries: pd.DataFrame, items: pd.DataFrame
) -> list[JudgePair]:
    texts = queries.set_index("query_id")["text"]
    rows = items.set_index("item_id", drop=False)
    return [
        JudgePair(qid, iid, texts[qid], item_card(rows.loc[iid]))
        for qid, iid in dict.fromkeys(pairs)
    ]


def judge_one(
    client: LLMClient,
    pair: JudgePair,
    model: str,
    lang: Lang,
    sleep: Callable[[float], None] = time.sleep,
) -> Judgment:
    error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            sleep(min(2.0**attempt, 30.0))
        try:
            parsed = client.chat_json(
                model,
                judge_messages(pair.query, pair.card, lang),
                tag=f"judge:{model}:{lang}",
                max_tokens=120,
                cache_extra={"rubric_version": RUBRIC_VERSION, "prompt_lang": lang},
                validate=Judgment.model_validate,
            )
            return Judgment.model_validate(parsed)
        except (LLMResponseError, ValidationError, openai.APIError) as e:
            error = e
    raise LLMResponseError(f"judge failed after {MAX_ATTEMPTS} attempts: {error}") from error


def judge_pairs(
    client: LLMClient,
    pairs: list[JudgePair],
    model: str,
    lang: Lang,
    workers: int = 8,
    seed: int = SEED,
    sleep: Callable[[float], None] = time.sleep,
) -> JudgeResult:
    order = list(pairs)
    random.Random(seed).shuffle(order)

    def run(pair: JudgePair) -> tuple[JudgePair, Judgment | str]:
        try:
            return pair, judge_one(client, pair, model, lang, sleep)
        except LLMResponseError as e:
            return pair, str(e)

    result = JudgeResult(model=model, lang=lang)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pair, outcome in pool.map(run, order):
            if isinstance(outcome, Judgment):
                result.grades.setdefault(pair.query_id, {})[pair.item_id] = outcome
            else:
                result.failures.append(
                    {"query_id": pair.query_id, "item_id": pair.item_id, "error": outcome}
                )
    result.grades = {
        qid: dict(sorted(items.items())) for qid, items in sorted(result.grades.items())
    }
    return result


def judgments_path(directory: Path, model: str, lang: Lang, subset: str) -> Path:
    return directory / f"{model}.{RUBRIC_VERSION}.{lang}.{subset}.json"


def save_judgments(result: JudgeResult, path: Path, subset: str) -> None:
    payload = {
        "meta": {
            "model": result.model,
            "rubric_version": RUBRIC_VERSION,
            "prompt_lang": result.lang,
            "subset": subset,
            "n_judged": result.n_judged,
            "n_failed": len(result.failures),
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "grades": {
            qid: {iid: j.model_dump() for iid, j in items.items()}
            for qid, items in result.grades.items()
        },
        "failures": result.failures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_judgments(path: Path) -> JudgeResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = JudgeResult(model=raw["meta"]["model"], lang=raw["meta"]["prompt_lang"])
    result.grades = {
        qid: {iid: Judgment.model_validate(j) for iid, j in items.items()}
        for qid, items in raw["grades"].items()
    }
    result.failures = raw["failures"]
    return result


def grades_of(result: JudgeResult) -> Qrels:
    return {qid: {iid: j.grade for iid, j in items.items()} for qid, items in result.grades.items()}


def write_qrels(result: JudgeResult, path: Path) -> Qrels:
    if result.model == RERANKER_MODEL:
        raise ValueError(f"{RERANKER_MODEL} is the reranker and can never produce qrels")
    qrels = grades_of(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qrels, ensure_ascii=False, indent=1), encoding="utf-8")
    return qrels


def load_qrels(path: Path) -> Qrels:
    raw: Mapping[str, Mapping[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return {qid: {iid: int(g) for iid, g in items.items()} for qid, items in raw.items()}
