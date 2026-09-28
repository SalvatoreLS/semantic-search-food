import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from foodsearch.cards import card_fields
from foodsearch.eval.judge import SEED, Qrels
from foodsearch.eval.pooling import Pool

HEAD_RANKS = 3
HEAD_PAIRS = 8
TAIL_PAIRS = 7
REPEATS = 30
LABEL_COLUMNS = ["query_id", "item_id", "grade", "annotator", "pass", "timestamp"]
QUEUE_COLUMNS = ["pair_id", "query_id", "item_id", "pass", "best_rank"]


@dataclass(frozen=True, slots=True)
class LabelPair:
    query_id: str
    item_id: str
    pass_: int
    best_rank: int

    @property
    def pair_id(self) -> str:
        return f"{self.query_id}:{self.item_id}:{self.pass_}"


def parse_pair_id(pair_id: str) -> tuple[str, str, int]:
    query_id, _, rest = pair_id.partition(":")
    item_id, _, pass_ = rest.rpartition(":")
    if not item_id or pass_ not in ("1", "2"):
        raise ValueError(f"Invalid pass in pair_id {pair_id!r}")
    return query_id, item_id, int(pass_)


def load_label_query_ids(path: Path) -> list[str]:
    return pd.read_csv(path, dtype=str)["query_id"].tolist()


def sample_pairs(pool: Pool, query_ids: Sequence[str], seed: int = SEED) -> list[LabelPair]:
    rng = random.Random(seed)
    target = HEAD_PAIRS + TAIL_PAIRS
    pairs: list[LabelPair] = []
    for qid in query_ids:
        ranks = pool.best_rank.get(qid, {})
        head = sorted(iid for iid, r in ranks.items() if r <= HEAD_RANKS)
        tail = sorted(iid for iid, r in ranks.items() if r > HEAD_RANKS)
        n_head = min(len(head), HEAD_PAIRS)
        n_tail = min(len(tail), target - n_head)
        n_head = min(len(head), target - n_tail)
        for iid in rng.sample(head, n_head) + rng.sample(tail, n_tail):
            pairs.append(LabelPair(qid, iid, 1, ranks[iid]))
    return pairs


def build_label_queue(
    pairs: Sequence[LabelPair], seed: int = SEED, repeats: int = REPEATS
) -> list[LabelPair]:
    rng = random.Random(seed)
    first = list(pairs)
    rng.shuffle(first)
    again = rng.sample(first, min(repeats, len(first)))
    return first + [LabelPair(p.query_id, p.item_id, 2, p.best_rank) for p in again]


def save_queue(queue: Sequence[LabelPair], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} is fixed once built; delete it to rebuild")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(p.pair_id, p.query_id, p.item_id, p.pass_, p.best_rank) for p in queue]
    pd.DataFrame(rows, columns=QUEUE_COLUMNS).to_csv(path, index=False)


def load_queue(path: Path) -> list[LabelPair]:
    df = pd.read_csv(path, dtype={"query_id": str, "item_id": str})
    return [
        LabelPair(r["query_id"], r["item_id"], int(r["pass"]), int(r["best_rank"]))
        for r in df.to_dict("records")
    ]


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(path, dtype={"query_id": str, "item_id": str, "annotator": str})


def record_label(path: Path, pair_id: str, grade: int, annotator: str) -> pd.DataFrame:
    if grade not in range(4):
        raise ValueError(f"Grade must be 0-3, got {grade}")
    query_id, item_id, pass_ = parse_pair_id(pair_id)
    labels = load_labels(path)
    same = (
        (labels["query_id"] == query_id)
        & (labels["item_id"] == item_id)
        & (labels["pass"] == pass_)
        & (labels["annotator"] == annotator)
    )
    row = {
        "query_id": query_id,
        "item_id": item_id,
        "grade": grade,
        "annotator": annotator,
        "pass": pass_,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    labels = pd.concat([labels[~same], pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    return labels


def human_qrels(labels: pd.DataFrame) -> Qrels:
    qrels: Qrels = {}
    for r in labels[labels["pass"] == 1].itertuples(index=False):
        qrels.setdefault(r.query_id, {})[r.item_id] = int(r.grade)
    return qrels


def intra_pairs(labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["query_id", "item_id", "annotator"]
    first = labels[labels["pass"] == 1][[*keys, "grade"]]
    second = labels[labels["pass"] == 2][[*keys, "grade"]]
    return first.merge(second, on=keys, suffixes=("_1", "_2"))


def export_sheet(
    queue: Sequence[LabelPair], queries: pd.DataFrame, items: pd.DataFrame, path: Path
) -> None:
    texts = queries.set_index("query_id")["text"]
    rows = items.set_index("item_id", drop=False)
    records = []
    for p in queue:
        card = card_fields(rows.loc[p.item_id])
        records.append(
            {
                "pair_id": p.pair_id,
                "query": texts[p.query_id],
                "name": card["name"],
                "category_path": card["category_path"],
                "description": card["description"],
                "price": card["price"],
                "attributes": ", ".join(card["attributes"]),
                "grade": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def import_sheet(sheet: Path, labels_path: Path, annotator: str) -> int:
    df = pd.read_csv(sheet, dtype={"pair_id": str})
    graded = df.dropna(subset=["grade"])
    for r in graded.itertuples(index=False):
        record_label(labels_path, r.pair_id, int(r.grade), annotator)
    return len(graded)
