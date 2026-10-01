import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from foodsearch.eval.judge import Qrels
from foodsearch.eval.pooling import ranked
from foodsearch.eval.stats import bootstrap_ci, holm, paired_randomization_test
from foodsearch.runs import Run

RELEVANT = 2
DEPTH = 10
LEAK_DEPTH = 5
METRICS = ["ndcg5", "ndcg10", "p5", "mrr", "food_leak5", "judged10"]
RANX_METRICS = {
    "ndcg5": "ndcg@5",
    "ndcg10": "ndcg@10",
    "p5": f"precision@5-l{RELEVANT}",
    "mrr": f"mrr@{DEPTH}-l{RELEVANT}",
}
PRODUCT_TYPE = "product"


def _ranx_scores(run: Run, qrels: Qrels) -> dict[str, np.ndarray]:
    from ranx import Qrels as RanxQrels
    from ranx import Run as RanxRun
    from ranx import evaluate

    ordered = {
        qid: {iid: float(DEPTH - r) for r, iid in enumerate(ranked(run.get(qid, {}), DEPTH))}
        for qid in qrels
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate(
            RanxQrels(qrels),
            RanxRun(ordered),
            list(RANX_METRICS.values()),
            return_mean=False,
            make_comparable=True,
        )


def per_query(
    runs: Mapping[str, Run],
    qrels: Qrels,
    is_food: Mapping[str, bool],
    query_types: Mapping[str, str],
) -> pd.DataFrame:
    query_ids = sorted(qrels)
    max_grade = np.array([max(qrels[q].values(), default=0) for q in query_ids])
    frames = []
    for system, run in runs.items():
        scores = _ranx_scores(run, qrels)
        frame = pd.DataFrame({"system": system, "query_id": query_ids})
        for name, metric in RANX_METRICS.items():
            frame[name] = scores[metric]
        frame.loc[max_grade == 0, ["ndcg5", "ndcg10"]] = np.nan
        frame.loc[max_grade < RELEVANT, "mrr"] = np.nan
        leak, judged = [], []
        for qid in query_ids:
            top = ranked(run.get(qid, {}), DEPTH)
            food_query = query_types.get(qid) != PRODUCT_TYPE
            non_food = sum(not is_food.get(iid, True) for iid in top[:LEAK_DEPTH])
            leak.append(non_food / LEAK_DEPTH if food_query else np.nan)
            judged.append(np.mean([iid in qrels[qid] for iid in top]) if top else np.nan)
        frame["food_leak5"] = leak
        frame["judged10"] = judged
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(table: pd.DataFrame, group: str = "system") -> pd.DataFrame:
    rows = []
    for key, sub in table.groupby(group, sort=False):
        for metric in METRICS:
            values = sub[metric].dropna().to_numpy()
            mean, lo, hi = bootstrap_ci(values)
            rows.append(
                {group: key, "metric": metric, "mean": mean, "lo": lo, "hi": hi, "n": len(values)}
            )
    return pd.DataFrame(rows)


def per_type(table: pd.DataFrame, query_tags: pd.DataFrame) -> pd.DataFrame:
    tags = query_tags.set_index("query_id")
    tagged = table.join(tags[["type", "us_translated"]], on="query_id")
    by_type = tagged.assign(group=tagged["type"])
    translated = tagged[tagged["us_translated"] == 1].assign(group="us_translated")
    frames = []
    for system, sub in pd.concat([by_type, translated]).groupby("system", sort=False):
        summary = summarize(sub, group="group")
        frames.append(summary.assign(system=system))
    return pd.concat(frames, ignore_index=True)[
        ["system", "group", "metric", "mean", "lo", "hi", "n"]
    ]


def comparison_pairs(systems: Sequence[str], against: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for base in (a for a in against if a in systems):
        for system in systems:
            if system != base and (base, system) not in pairs:
                pairs.append((system, base))
    return pairs


def compare(
    table: pd.DataFrame, pairs: Sequence[tuple[str, str]], metrics: Sequence[str] = METRICS[:5]
) -> pd.DataFrame:
    wide = {m: table.pivot(index="query_id", columns="system", values=m) for m in metrics}
    rows = []
    for metric in metrics:
        family = []
        for a, b in pairs:
            both = wide[metric][[a, b]].dropna()
            family.append(
                {
                    "metric": metric,
                    "system_a": a,
                    "system_b": b,
                    "n": len(both),
                    "mean_a": both[a].mean(),
                    "mean_b": both[b].mean(),
                    "diff": both[a].mean() - both[b].mean(),
                    "p": paired_randomization_test(both[a], both[b]),
                }
            )
        for row, adjusted in zip(family, holm([r["p"] for r in family]), strict=True):
            row["p_holm"] = adjusted
            row["significant"] = adjusted < 0.05
        rows += family
    return pd.DataFrame(rows)


def gap_counts(qrels: Qrels) -> dict[str, int]:
    max_grade = [max(items.values(), default=0) for items in qrels.values()]
    return {
        "queries": len(max_grade),
        "no_graded_item": sum(g == 0 for g in max_grade),
        "no_relevant_item": sum(g < RELEVANT for g in max_grade),
    }
