import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from foodsearch.eval.agreement import AgreementReport
from foodsearch.eval.human import LabelPair
from foodsearch.eval.pooling import Pool
from foodsearch.eval.prompts import RUBRIC_VERSION, Lang, few_shot_hash, system_prompt_hash
from foodsearch.llm import RERANKER_MODEL, text_hash


def write_freeze(
    path: Path,
    *,
    judge_model: str,
    lang: Lang,
    report: AgreementReport,
    queue: Sequence[LabelPair],
    pool: Pool,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"{path} already exists; the evaluation is frozen")
    if judge_model == RERANKER_MODEL:
        raise ValueError(f"{RERANKER_MODEL} is the reranker and cannot be the frozen judge")
    if not report.gate_passed:
        raise ValueError("Agreement gate not passed; revise the rubric before freezing")
    freeze = {
        "rubric_version": RUBRIC_VERSION,
        "prompt_lang": lang,
        "judge_model": judge_model,
        "system_prompt_sha256": system_prompt_hash(lang),
        "few_shot_sha256": few_shot_hash(lang),
        "human_label_query_ids": sorted({p.query_id for p in queue}),
        "label_pairs_sha256": text_hash("\n".join(p.pair_id for p in queue)),
        "agreement": report.to_dict(),
        "pool_depth": pool.depth,
        "pool_systems": pool.systems,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(freeze, ensure_ascii=False, indent=1), encoding="utf-8")
    return freeze


def load_freeze(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_prompt_unchanged(freeze: dict[str, Any]) -> None:
    lang = freeze["prompt_lang"]
    current = (RUBRIC_VERSION, system_prompt_hash(lang), few_shot_hash(lang))
    frozen = (freeze["rubric_version"], freeze["system_prompt_sha256"], freeze["few_shot_sha256"])
    if current != frozen:
        raise ValueError("Judge prompt or rubric changed after the freeze; bump RUBRIC_VERSION")
