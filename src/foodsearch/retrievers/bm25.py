import bm25s
import numpy as np
import pandas as pd

from foodsearch.retrievers.base import Hit
from foodsearch.text import tokenize


class BM25Retriever:
    def __init__(
        self,
        name: str = "bm25",
        text_column: str = "doc",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.name = name
        self.text_column = text_column
        self.k1 = k1
        self.b = b
        self._index: bm25s.BM25 | None = None
        self._item_ids: np.ndarray = np.array([], dtype=object)

    def fit(self, items: pd.DataFrame) -> None:
        corpus = [tokenize(text) for text in items[self.text_column]]
        self._index = bm25s.BM25(k1=self.k1, b=self.b)
        self._index.index(corpus, show_progress=False)
        self._item_ids = items["item_id"].to_numpy(dtype=object)

    def search(self, query: str, k: int = 100) -> list[Hit]:
        if self._index is None:
            raise RuntimeError(f"{self.name} must be fitted before search")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._index.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Hit(item_id=str(self._item_ids[i]), score=float(scores[i]), source=self.name)
            for i in order
            if scores[i] > 0
        ]
