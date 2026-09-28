from pathlib import Path

import pytest

from foodsearch.eval.agreement import AgreementReport
from foodsearch.eval.freeze import check_prompt_unchanged, load_freeze, write_freeze
from foodsearch.eval.human import LabelPair
from foodsearch.eval.pooling import Pool

POOL = Pool(depth=10, systems=["bm25"], best_rank={"q001": {"i1": 1}})
QUEUE = [LabelPair("q001", "i1", 1, 1)]


def _report(passed: bool) -> AgreementReport:
    return AgreementReport(1, 1, {}, {}, [[1]], passed)


def test_freeze_writes_once_and_checks_prompt(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    assert load_freeze(path) is None
    kwargs = {"judge_model": "gpt-4.1", "lang": "pt", "queue": QUEUE, "pool": POOL}
    write_freeze(path, report=_report(True), **kwargs)
    freeze = load_freeze(path)
    assert freeze is not None and freeze["human_label_query_ids"] == ["q001"]
    check_prompt_unchanged(freeze)
    with pytest.raises(FileExistsError):
        write_freeze(path, report=_report(True), **kwargs)
    with pytest.raises(ValueError, match="changed"):
        check_prompt_unchanged({**freeze, "few_shot_sha256": "0"})


def test_freeze_refuses_failed_gate_or_reranker(tmp_path: Path) -> None:
    common = {"lang": "en", "queue": QUEUE, "pool": POOL}
    with pytest.raises(ValueError, match="gate"):
        write_freeze(tmp_path / "a.json", judge_model="gpt-4.1", report=_report(False), **common)
    with pytest.raises(ValueError, match="reranker"):
        write_freeze(
            tmp_path / "b.json", judge_model="gpt-4.1-mini", report=_report(True), **common
        )
