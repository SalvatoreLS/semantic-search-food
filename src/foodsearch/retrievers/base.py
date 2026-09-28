from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True, slots=True)
class Hit:
    item_id: str
    score: float
    source: str


class Retriever(Protocol):
    name: str

    def fit(self, items: pd.DataFrame) -> None: ...

    def search(self, query: str, k: int = 100) -> list[Hit]: ...
