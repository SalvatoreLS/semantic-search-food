import json
from collections.abc import Mapping, Sequence
from pathlib import Path

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
