from collections.abc import Mapping

import pandas as pd

from foodsearch.eval.judge import Qrels
from foodsearch.eval.pooling import ranked
from foodsearch.runs import Run

RELEVANT = 2
PERFECT = 3
TOP = 5
WORST = 10


def failure_table(
    table: pd.DataFrame,
    qrels: Qrels,
    runs: Mapping[str, Run],
    tags: pd.DataFrame,
    system: str,
    n: int = WORST,
) -> pd.DataFrame:
    wide = table.pivot(index="query_id", columns="system", values="ndcg5")
    mine = table[table["system"] == system].set_index("query_id")
    worst = mine.sort_values(["ndcg5", "p5"], na_position="first").head(n)
    tag = tags.set_index("query_id")
    rows = []
    for qid, row in worst.iterrows():
        grades = qrels.get(qid, {})
        top = ranked(runs[system].get(qid, {}), TOP)
        others = wide.loc[qid].drop(system).dropna()
        relevant = sum(g >= RELEVANT for g in grades.values())
        rows.append(
            {
                "query_id": qid,
                "type": tag.at[qid, "type"],
                "us_translated": int(tag.at[qid, "us_translated"]),
                "ndcg5": row["ndcg5"],
                "p5": row["p5"],
                "food_leak5": row["food_leak5"],
                "pool_relevant": relevant,
                "pool_perfect": sum(g >= PERFECT for g in grades.values()),
                "top5_relevant": sum(grades.get(i, 0) >= RELEVANT for i in top),
                "catalog_gap": relevant < TOP,
                "best_system": others.idxmax() if len(others) else None,
                "best_ndcg5": others.max() if len(others) else None,
            }
        )
    return pd.DataFrame(rows)
