import numpy as np
import pytest

from foodsearch.eval.agreement import agreement, binary_kappa, cohen_kappa, confusion
from foodsearch.eval.stats import bootstrap_ci, cluster_bootstrap, holm, paired_randomization_test


def test_kappa_hand_computed_values() -> None:
    same = np.array([0, 1, 2, 3])
    assert cohen_kappa(same, same, "quadratic") == pytest.approx(1.0)
    assert cohen_kappa(np.array([0, 3]), np.array([3, 0]), "quadratic") == pytest.approx(-1.0)
    assert cohen_kappa(same, np.array([1, 1, 2, 3]), "linear") == pytest.approx(7 / 9)
    assert binary_kappa(np.array([0, 0, 2, 2]), np.array([0, 2, 2, 3])) == pytest.approx(0.5)
    assert np.isnan(cohen_kappa(np.array([2, 2]), np.array([2, 2]), "quadratic"))


def test_confusion_rows_are_human() -> None:
    matrix = confusion(np.array([0, 0, 3]), np.array([1, 1, 3]))
    assert matrix[0, 1] == 2 and matrix[3, 3] == 1 and matrix.sum() == 3


def test_agreement_report_and_gate() -> None:
    human = {f"q{i}": {"a": 0, "b": 1, "c": 2, "d": 3} for i in range(6)}
    report = agreement(human, human, n=200)
    assert report.gate_passed and report.n_pairs == 24 and report.n_queries == 6
    assert report.values["exact"] == 1.0 and report.values["mean_signed_diff"] == 0.0
    lenient = {q: {i: min(g + 1, 3) for i, g in items.items()} for q, items in human.items()}
    report = agreement(human, lenient, n=200)
    assert report.values["mean_signed_diff"] > 0
    assert report.values["within_1"] == 1.0
    assert agreement(human, lenient, n=200).ci == report.ci


def test_bootstrap_ci_brackets_the_mean_and_is_seeded() -> None:
    values = np.linspace(0, 1, 50)
    mean, lo, hi = bootstrap_ci(values)
    assert lo < mean < hi
    assert bootstrap_ci(values) == (mean, lo, hi)
    assert all(np.isnan(bootstrap_ci([])))


def test_cluster_bootstrap_resamples_whole_groups() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0])
    lo, hi = cluster_bootstrap(lambda idx: values[idx].mean(), ["a", "a", "b", "b"], n=500)
    assert 0.0 <= lo <= hi <= 1.0


def test_randomization_test() -> None:
    a = np.linspace(0.2, 0.8, 40)
    assert paired_randomization_test(a, a) == pytest.approx(1.0)
    assert paired_randomization_test(a + 0.1, a) < 0.001


def test_holm_known_values() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
