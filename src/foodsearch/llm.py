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
JUDGE_MODEL = "gpt-4.1"
RERANKER_MODEL = "gpt-4.1-mini"
UNDERSTANDING_MODEL = "gpt-4.1-mini"

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
    """Look up a model's price per million tokens. A model missing from PRICES raises an error here,
    before any request is sent, so no call can go out unpriced.
    """
    if model not in PRICES:
        raise KeyError(f"No price recorded for model {model!r}; add it to PRICES first")
    return PRICES[model]


def call_cost(model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call from its token counts. Input tokens that OpenAI served from its
    prompt cache are billed at the cheaper cached rate.
    """
    price = price_of(model)
    fresh = input_tokens - cached_input_tokens
    total = fresh * price.input + cached_input_tokens * price.cached_input
    return (total + output_tokens * price.output) / 1_000_000


def cache_key(payload: dict[str, Any]) -> str:
    """Stable sha256 of a JSON payload, with sorted keys so the same request always gets the same
    key. Every chat and embedding entry in the disk cache is stored under one.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    """sha256 of a text. Used for the embedding cache keys and for the prompt hashes that the
    evaluation freeze checks.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CostTracker:
    """Keeps every paid call of this process in memory and appends it to artifacts/cost_log.jsonl,
    the file the README spend and the cost report come from.
    """

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
        """Price one paid call and log it. A lock guards the list and the file, because the judge
        and the pointwise reranker call the API from thread pools.
        """
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
        """Dollars spent by this process so far."""
        return sum(c.usd for c in self.calls)


def load_cost_log(log_path: Path) -> pd.DataFrame:
    """Read the cost log into a DataFrame, or an empty one before the first paid call."""
    columns = list(CallCost.__dataclass_fields__)
    if not log_path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_json(log_path, lines=True, dtype={"tag": str, "model": str})


def cost_summary(log_path: Path) -> pd.DataFrame:
    """Tokens and dollars per (tag, model), for example how much the judge or the listwise rerank
    cost over the whole project.
    """
    log = load_cost_log(log_path)
    numeric = ["input_tokens", "cached_input_tokens", "output_tokens", "usd"]
    return log.groupby(["tag", "model"], as_index=False)[numeric].sum()


class LLMClient:
    """The only way the project talks to OpenAI.

    Every chat and embedding call goes through here. Calls run at temperature 0, are cached on disk
    in artifacts/cache and, when they actually reach the API, are appended to the cost log. Reruns
    are therefore free and reproducible, and the spend reported in the README is exact.
    """

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
        """The OpenAI client, built on first use from the key in .env. Cached runs and the tests
        never touch it, so they work without a key.
        """
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
        """Send one chat request that must answer with a JSON object, and cache the answer.

        The cache key covers the model, the full messages, the call parameters and any `cache_extra`
        (the judge adds its rubric version and prompt language there). An answer is stored only
        after it finished normally, parsed as a JSON object and passed `validate`, so a truncated or
        malformed answer is never cached and the next attempt really calls the API again. Query
        understanding, both LLM rerankers and the judge all go through here, usually via
        chat_checked.
        """
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
        """Embed texts with an OpenAI embedding model, one cached vector per text.

        Only texts missing from the cache are sent, in batches of 256, so embedding the catalog is a
        one-off cost and every later query embedding is a cache lookup when the query was seen
        before. This backs the OpenAI side of dense retrieval (stage 2).
        """
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
        sleep: Callable[[float], None] | None = None,
    ) -> T:
        """chat_json with parsing and retries: the safe way to get a structured answer.

        `parse` turns the JSON into the caller's type and rejects bad answers (a grade out of range,
        a ranking with repeated numbers). Invalid answers and API errors are retried with
        exponential backoff, and LLMResponseError is raised after the last attempt. Each caller
        picks its own fallback: the pipeline keeps the order it already had, and the judge leaves
        the pair unjudged instead of grading it 0.
        """
        error: Exception | None = None
        for attempt in range(attempts):
            if attempt:
                (sleep or time.sleep)(min(2.0**attempt, 30.0))
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
        """Dollars spent after the first `n_calls` calls of this client. The pipeline notes the
        call count when a stage starts and uses this to attribute cost to that stage.
        """
        return sum(c.usd for c in self.costs.calls[n_calls:])


@cache
def default_client() -> LLMClient:
    """The shared client over artifacts/cache and artifacts/cost_log.jsonl, used whenever no client
    is passed in explicitly.
    """
    return LLMClient(paths.cache_dir(), paths.cost_log())
