from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from foodsearch.retrievers import DenseRetriever, Hit, weighted_rrf
from foodsearch.retrievers.dense import SentenceTransformerEncoder


def test_openai_dense_ranks_by_cosine(make_llm: Any, items: pd.DataFrame) -> None:
    llm, fake = make_llm(lambda messages: "{}")
    dense = DenseRetriever(name="d", backend="openai", model="text-embedding-3-small", llm=llm)
    dense.fit(items)
    hits = dense.search("pizza grande", k=2)
    assert [h.item_id for h in hits] == ["pizza", hits[1].item_id] and len(hits) == 2
    assert all(h.source == "d" for h in hits)
    assert hits[0].score > hits[1].score
    dense.fit(items)
    assert sum(len(batch) for batch in fake.embed_calls) == len(items) + 1


class FakeSentenceModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.inputs.append(texts)
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_st_encoder_prefixes_and_caches_passages(tmp_path: Path, items: pd.DataFrame) -> None:
    model = FakeSentenceModel()
    dense = DenseRetriever(
        backend="st",
        model="org/e5",
        query_prefix="query: ",
        passage_prefix="passage: ",
        cache_dir=tmp_path,
    )
    encoder = dense.encoder
    assert isinstance(encoder, SentenceTransformerEncoder)
    encoder._model = model
    dense.fit(items)
    dense.fit(items)
    dense.search("pizza")
    assert len(model.inputs) == 2
    assert all(t.startswith("passage: ") for t in model.inputs[0])
    assert model.inputs[1] == ["query: pizza"]
    assert len(list(tmp_path.glob("org_e5.*.npy"))) == 1


def test_weighted_rrf_scores_weights_and_sources() -> None:
    a = [Hit("x", 0.9, "a"), Hit("y", 0.8, "a")]
    b = [Hit("y", 5.0, "b"), Hit("z", 4.0, "b")]
    fused = weighted_rrf({"dense": a, "bm25": b}, {"bm25": 0.5}, k=60, source="s")
    scores = {h.item_id: h.score for h in fused.hits}
    assert scores["x"] == 1 / 61
    assert scores["y"] == 1 / 62 + 0.5 / 61
    assert scores["z"] == 0.5 / 62
    assert [h.item_id for h in fused.hits] == ["y", "x", "z"]
    assert all(h.source == "s" for h in fused.hits)
    assert fused.sources["y"] == [("dense", 2), ("bm25", 1)]


def test_rrf_ties_break_by_item_id() -> None:
    fused = weighted_rrf({"a": [Hit("b", 1.0, "a")], "c": [Hit("a", 1.0, "c")]})
    assert [h.item_id for h in fused.hits] == ["a", "b"]
