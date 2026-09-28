from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from foodsearch.eval.judge import SEED, Qrels
from foodsearch.eval.stats import N_BOOTSTRAP, cluster_bootstrap

GRADES = 4
RELEVANT = 2
GATE = 0.6


def confusion(human: np.ndarray, judge: np.ndarray, k: int = GRADES) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=int)
    np.add.at(matrix, (human, judge), 1)
    return matrix


def cohen_kappa(
    human: np.ndarray, judge: np.ndarray, weights: str | None = None, k: int = GRADES
) -> float:
    observed = confusion(human, judge, k).astype(float)
    total = observed.sum()
    if total == 0:
        return np.nan
    observed /= total
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    i, j = np.indices((k, k))
    if weights == "quadratic":
        disagreement = (i - j) ** 2 / (k - 1) ** 2
    elif weights == "linear":
        disagreement = np.abs(i - j) / (k - 1)
    else:
        disagreement = (i != j).astype(float)
    expected_disagreement = (disagreement * expected).sum()
    if expected_disagreement == 0:
        return np.nan
    return float(1 - (disagreement * observed).sum() / expected_disagreement)


def binary_kappa(human: np.ndarray, judge: np.ndarray) -> float:
    return cohen_kappa((human >= RELEVANT).astype(int), (judge >= RELEVANT).astype(int), k=2)


STATISTICS = {
    "kappa_quadratic": lambda h, j: cohen_kappa(h, j, "quadratic"),
    "kappa_linear": lambda h, j: cohen_kappa(h, j, "linear"),
    "kappa_unweighted": lambda h, j: cohen_kappa(h, j),
    "kappa_binary": binary_kappa,
    "exact": lambda h, j: float(np.mean(h == j)),
    "within_1": lambda h, j: float(np.mean(np.abs(h - j) <= 1)),
    "mean_signed_diff": lambda h, j: float(np.mean(j - h)),
}


@dataclass
class AgreementReport:
    n_pairs: int
    n_queries: int
    values: dict[str, float]
    ci: dict[str, tuple[float, float]]
    confusion: list[list[int]]
    gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_grades(human: Qrels, judge: Qrels) -> pd.DataFrame:
    rows = [
        (qid, iid, grade, judge[qid][iid])
        for qid, items in human.items()
        for iid, grade in items.items()
        if iid in judge.get(qid, {})
    ]
    return pd.DataFrame(rows, columns=["query_id", "item_id", "human", "judge"])


def agreement(
    human: Qrels, judge: Qrels, n: int = N_BOOTSTRAP, seed: int = SEED
) -> AgreementReport:
    pairs = paired_grades(human, judge)
    h = pairs["human"].to_numpy(dtype=int)
    j = pairs["judge"].to_numpy(dtype=int)
    groups = pairs["query_id"].to_numpy()
    values = {name: fn(h, j) for name, fn in STATISTICS.items()}
    ci = {
        name: cluster_bootstrap(lambda idx, fn=fn: fn(h[idx], j[idx]), groups, n, seed)
        for name, fn in STATISTICS.items()
    }
    passed = values["kappa_quadratic"] >= GATE and values["kappa_binary"] >= GATE
    return AgreementReport(
        n_pairs=len(pairs),
        n_queries=int(pairs["query_id"].nunique()),
        values=values,
        ci=ci,
        confusion=confusion(h, j).tolist(),
        gate_passed=bool(passed),
    )


def intra_agreement(pairs: pd.DataFrame) -> dict[str, float]:
    first = pairs["grade_1"].to_numpy(dtype=int)
    second = pairs["grade_2"].to_numpy(dtype=int)
    return {
        "n_pairs": len(pairs),
        "exact": float(np.mean(first == second)) if len(pairs) else np.nan,
        "kappa_quadratic": cohen_kappa(first, second, "quadratic") if len(pairs) else np.nan,
    }
