from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from foodsearch.eval.judge import Qrels
from foodsearch.eval.metrics import METRICS
from foodsearch.eval.stats import ALPHA, bootstrap_ci, holm, paired_randomization_test


@dataclass(frozen=True, slots=True)
class Step:
    family: str
    system: str
    base: str
    label: str


STEPS = (
    Step("backend", "dense_oai_large", "dense_oai_small", "text-embedding-3-large vs -small"),
    Step("backend", "dense_oai_large", "dense_e5_base", "text-embedding-3-large vs e5-base"),
    Step("backend", "dense_oai_large", "dense_bge_m3", "text-embedding-3-large vs bge-m3"),
    Step("component", "dense_prior", "dense_oai_large", "+ intent food prior"),
    Step("component", "qu_fusion", "dense_prior", "+ LLM query understanding, raw/expanded RRF"),
    Step("component", "qu_fusion_ce", "qu_fusion", "+ bge cross-encoder rerank"),
    Step("component", "main", "qu_fusion", "+ LLM listwise rerank"),
    Step("component", "hybrid", "main", "+ BM25 in weighted RRF"),
)


def ablations(
    table: pd.DataFrame, steps: Sequence[Step] = STEPS, metrics: Sequence[str] = METRICS[:5]
) -> pd.DataFrame:
    systems = set(table["system"])
    usable = [s for s in steps if s.system in systems and s.base in systems]
    rows = []
    for metric in metrics:
        wide = table.pivot(index="query_id", columns="system", values=metric)
        for family in dict.fromkeys(s.family for s in usable):
            group = []
            for step in (s for s in usable if s.family == family):
                both = wide[[step.system, step.base]].dropna()
                diff, lo, hi = bootstrap_ci(both[step.system] - both[step.base])
                group.append(
                    {
                        "family": family,
                        "label": step.label,
                        "system": step.system,
                        "base": step.base,
                        "metric": metric,
                        "n": len(both),
                        "diff": diff,
                        "lo": lo,
                        "hi": hi,
                        "ci_excludes_0": bool(lo > 0 or hi < 0),
                        "p": paired_randomization_test(both[step.system], both[step.base]),
                    }
                )
            for row, adjusted in zip(group, holm([r["p"] for r in group]), strict=True):
                row["p_holm"] = adjusted
                row["significant"] = adjusted < ALPHA
            rows += group
    return pd.DataFrame(rows)


RERANK_STEPS = (
    Step("rerank", "main", "qu_fusion", "LLM listwise rerank"),
    Step("rerank", "qu_fusion_ce", "qu_fusion", "bge cross-encoder rerank"),
    Step("rerank", "dense_pointwise", "dense_oai_large", "LLM pointwise rerank"),
)


def restrict(qrels: Qrels, to: Qrels) -> Qrels:
    return {
        qid: {iid: grade for iid, grade in qrels.get(qid, {}).items() if iid in items}
        for qid, items in to.items()
    }


def rerank_gain(
    judge_table: pd.DataFrame, human_table: pd.DataFrame, steps: Sequence[Step] = RERANK_STEPS
) -> pd.DataFrame:
    columns = ["label", "system", "base", "n", "diff", "lo", "hi", "p_holm"]
    judged = ablations(judge_table, steps, ["ndcg5"])[columns]
    human = ablations(human_table, steps, ["ndcg5"])[columns]
    return judged.merge(human, on=["label", "system", "base"], suffixes=("_judge", "_human"))
