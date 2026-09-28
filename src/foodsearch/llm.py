import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

import diskcache
import numpy as np
import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from foodsearch import paths

EMBED_BATCH = 256

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Price:
    input: float
    cached_input: float
    output: float


PRICES: dict[str, Price] = {
    "gpt-4.1": Price(2.00, 0.50, 8.00),
    "gpt-4.1-mini": Price(0.40, 0.10, 1.60),
    "gpt-4.1-nano": Price(0.10, 0.025, 0.40),
    "text-embedding-3-small": Price(0.02, 0.02, 0.0),
    "text-embedding-3-large": Price(0.13, 0.13, 0.0),
}


class LLMResponseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CallCost:
    ts: str
    tag: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    usd: float


def price_of(model: str) -> Price:
    if model not in PRICES:
        raise KeyError(f"No price recorded for model {model!r}; add it to PRICES first")
    return PRICES[model]


def call_cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    price = price_of(model)
    fresh = input_tokens - cached_input_tokens
    total = fresh * price.input + cached_input_tokens * price.cached_input
    return (total + output_tokens * price.output) / 1_000_000


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CostTracker:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.calls: list[CallCost] = []
        self._lock = threading.Lock()

    def record(
        self,
        tag: str,
        model: str,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> CallCost:
        cost = CallCost(
            ts=datetime.now(UTC).isoformat(timespec="seconds"),
            tag=tag,
            model=model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            usd=call_cost(model, input_tokens, cached_input_tokens, output_tokens),
        )
        with self._lock:
            self.calls.append(cost)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(cost)) + "\n")
        return cost

    @property
    def total_usd(self) -> float:
        return sum(c.usd for c in self.calls)


def load_cost_log(log_path: Path) -> pd.DataFrame:
    columns = list(CallCost.__dataclass_fields__)
    if not log_path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_json(log_path, lines=True, dtype={"tag": str, "model": str})


def cost_summary(log_path: Path) -> pd.DataFrame:
    log = load_cost_log(log_path)
    numeric = ["input_tokens", "cached_input_tokens", "output_tokens", "usd"]
    return log.groupby(["tag", "model"], as_index=False)[numeric].sum()


class LLMClient:
    def __init__(
        self,
        cache_dir: Path,
        cost_log_path: Path,
        client: Any | None = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.cache = diskcache.Cache(str(cache_dir))
        self.costs = CostTracker(cost_log_path)
        self._client = client
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def client(self) -> Any:
        if self._client is None:
            load_dotenv()
            self._client = OpenAI(max_retries=self._max_retries, timeout=self._timeout)
        return self._client

    def chat_json(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        *,
        tag: str,
        max_tokens: int | None = None,
        cache_extra: dict[str, Any] | None = None,
        validate: Callable[[dict[str, Any]], object] | None = None,
    ) -> dict[str, Any]:
        price_of(model)
        params: dict[str, Any] = {"temperature": 0, "response_format": {"type": "json_object"}}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        key = cache_key(
            {
                "kind": "chat",
                "model": model,
                "messages": list(messages),
                "params": params,
                "extra": cache_extra or {},
            }
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        response = self.client.chat.completions.create(
            model=model, messages=list(messages), **params
        )
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        self.costs.record(
            tag,
            model,
            input_tokens=usage.prompt_tokens,
            cached_input_tokens=getattr(details, "cached_tokens", None) or 0,
            output_tokens=usage.completion_tokens,
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise LLMResponseError(f"{model} stopped with finish_reason={choice.finish_reason!r}")
        try:
            parsed = json.loads(choice.message.content or "")
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"{model} returned invalid JSON") from e
        if not isinstance(parsed, dict):
            raise LLMResponseError(f"{model} returned JSON that is not an object")
        if validate is not None:
            validate(parsed)
        self.cache.set(key, parsed)
        return parsed

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        tag: str,
        dimensions: int | None = None,
    ) -> np.ndarray:
        price_of(model)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        def key_of(text: str) -> str:
            return cache_key(
                {"kind": "embed", "model": model, "dimensions": dimensions, "text": text_hash(text)}
            )

        missing = list(dict.fromkeys(t for t in texts if key_of(t) not in self.cache))
        extra = {"dimensions": dimensions} if dimensions is not None else {}
        for start in range(0, len(missing), EMBED_BATCH):
            batch = missing[start : start + EMBED_BATCH]
            response = self.client.embeddings.create(model=model, input=batch, **extra)
            self.costs.record(tag, model, input_tokens=response.usage.prompt_tokens)
            data = sorted(response.data, key=lambda d: d.index)
            for text, item in zip(batch, data, strict=True):
                self.cache.set(key_of(text), np.asarray(item.embedding, dtype=np.float32))
        return np.stack([self.cache[key_of(t)] for t in texts])

    def chat_checked(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        parse: Callable[[dict[str, Any]], T],
        *,
        tag: str,
        attempts: int = 3,
        max_tokens: int | None = None,
        cache_extra: dict[str, Any] | None = None,
    ) -> T:
        error: Exception | None = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(min(2.0**attempt, 30.0))
            try:
                parsed = self.chat_json(
                    model,
                    messages,
                    tag=tag,
                    max_tokens=max_tokens,
                    cache_extra=cache_extra,
                    validate=parse,
                )
                return parse(parsed)
            except (LLMResponseError, ValueError, openai.APIError) as e:
                error = e
        raise LLMResponseError(f"{model} failed after {attempts} attempts: {error}") from error

    def cost_since(self, n_calls: int) -> float:
        return sum(c.usd for c in self.costs.calls[n_calls:])


@cache
def default_client() -> LLMClient:
    return LLMClient(paths.cache_dir(), paths.cost_log())
