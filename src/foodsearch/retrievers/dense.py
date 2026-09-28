import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from foodsearch import paths
from foodsearch.device import resolve_device
from foodsearch.llm import LLMClient, default_client
from foodsearch.retrievers.base import Hit

Backend = Literal["openai", "st"]
TextKind = Literal["query", "passage"]


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray: ...


class OpenAIEncoder:
    def __init__(self, model: str, dimensions: int | None, llm: LLMClient | None) -> None:
        self.model = model
        self.dimensions = dimensions
        self._llm = llm

    def encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        llm = self._llm or default_client()
        return llm.embed(self.model, texts, tag=f"embed:{kind}", dimensions=self.dimensions)


class SentenceTransformerEncoder:
    def __init__(
        self,
        model: str,
        query_prefix: str,
        passage_prefix: str,
        batch_size: int,
        cache_dir: Path | None,
    ) -> None:
        self.model = model
        self.prefixes: dict[TextKind, str] = {"query": query_prefix, "passage": passage_prefix}
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = resolve_device()
            self._model = SentenceTransformer(self.model, device=device)
            if device == "cuda":
                self._model.half()
        return self._model

    def _cache_path(self, texts: Sequence[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(self.prefixes["passage"].encode("utf-8"))
        for text in texts:
            digest.update(b"\x00" + text.encode("utf-8"))
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model)
        return (self.cache_dir or paths.embeddings_dir()) / f"{slug}.{digest.hexdigest()[:16]}.npy"

    def encode(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        cache_path = self._cache_path(texts) if kind == "passage" else None
        if cache_path is not None and cache_path.exists():
            return np.load(cache_path)
        vectors = (
            self._load()
            .encode(
                [self.prefixes[kind] + t for t in texts],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            .astype(np.float32)
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vectors)
        return vectors


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class DenseRetriever:
    def __init__(
        self,
        name: str = "dense",
        model: str = "text-embedding-3-large",
        backend: Backend = "openai",
        dimensions: int | None = None,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        text_column: str = "doc",
        llm: LLMClient | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.text_column = text_column
        self.encoder: Encoder
        if backend == "openai":
            self.encoder = OpenAIEncoder(model, dimensions, llm)
        elif backend == "st":
            self.encoder = SentenceTransformerEncoder(
                model, query_prefix, passage_prefix, batch_size, cache_dir
            )
        else:
            raise ValueError(f"Unknown dense backend {backend!r}")
        self._matrix: np.ndarray | None = None
        self._item_ids: np.ndarray = np.array([], dtype=object)

    def fit(self, items: pd.DataFrame) -> None:
        texts = items[self.text_column].tolist()
        self._matrix = normalize(self.encoder.encode(texts, "passage"))
        self._item_ids = items["item_id"].to_numpy(dtype=object)

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        return normalize(self.encoder.encode(queries, "query"))

    def search_vector(self, vector: np.ndarray, k: int = 100) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError(f"{self.name} must be fitted before search")
        scores = self._matrix @ vector
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        order = top[np.lexsort((self._item_ids[top].astype(str), -scores[top]))]
        return [
            Hit(item_id=str(self._item_ids[i]), score=float(scores[i]), source=self.name)
            for i in order
        ]

    def search(self, query: str, k: int = 100) -> list[Hit]:
        return self.search_vector(self.embed_queries([query])[0], k)
