from collections.abc import Callable, Sequence

import numpy as np

from foodsearch.eval.judge import SEED

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 10_000
ALPHA = 0.05


def bootstrap_ci(
    values: Sequence[float] | np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n, x.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return (float(x.mean()), float(lo), float(hi))


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    groups: Sequence[str] | np.ndarray,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float]:
    labels = np.asarray(groups)
    clusters = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for i in range(n):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        stats[i] = statistic(np.concatenate([clusters[p] for p in picks]))
    lo, hi = np.nanpercentile(stats, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return (float(lo), float(hi))


def paired_randomization_test(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0:
        return np.nan
    observed = abs(diff.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, diff.size))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed - 1e-12) + 1) / (n + 1))


def holm(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()
