import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from foodsearch.retrievers.base import Hit

Run = dict[str, dict[str, float]]


def hits_to_run(hits_by_query: Mapping[str, Sequence[Hit]]) -> Run:
    return {qid: {h.item_id: h.score for h in hits} for qid, hits in hits_by_query.items()}


def save_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")


def load_run(path: Path) -> Run:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {qid: {iid: float(s) for iid, s in scores.items()} for qid, scores in raw.items()}


FINAL_COLUMNS = ["query_id", "rank", "itemId", "score", "system"]


def top_k_table(run: Run, system: str, k: int = 10) -> pd.DataFrame:
    rows = [
        (qid, rank, iid, scores[iid], system)
        for qid, scores in sorted(run.items())
        for rank, iid in enumerate(sorted(scores, key=lambda i: (-scores[i], i))[:k], start=1)
    ]
    return pd.DataFrame(rows, columns=FINAL_COLUMNS)
