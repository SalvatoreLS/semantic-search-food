import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import openai
import pytest

from foodsearch import paths

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CONFIG = "bm25:\n  type: bm25\ndense_a:\n  type: dense\n"


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delitem(sys.modules, "build_artifacts", raising=False)
    import build_artifacts

    return build_artifacts


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path))
    paths.data_dir().mkdir()
    paths.items_csv().write_text("itemId\n", encoding="utf-8")
    paths.queries_csv().write_text("search_term_pt\n", encoding="utf-8")
    paths.configs_dir().mkdir()
    (paths.configs_dir() / "systems.yaml").write_text(CONFIG, encoding="utf-8")
    return tmp_path


class Recorder:
    def __init__(self, failing: dict[str, BaseException] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.failing = failing or {}

    def cli(self, argv: Sequence[str]) -> None:
        self.calls.append(list(argv))
        command = argv[0] if argv[0] != "run" else f"run {argv[2]}"
        if command in self.failing:
            raise self.failing[command]
        if argv[0] == "run":
            paths.runs_dir().mkdir(parents=True, exist_ok=True)
            (paths.runs_dir() / f"{argv[2]}.json").write_text("{}", encoding="utf-8")
        if argv[0] == "pool":
            paths.pool_json().write_text("{}", encoding="utf-8")

    def images(self, argv: Sequence[str]) -> int:
        self.calls.append(["images", *argv])
        return 0


def _patch(build: ModuleType, monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> None:
    monkeypatch.setattr(build.cli, "main", recorder.cli)
    monkeypatch.setattr(build.fetch_images, "main", recorder.images)


def test_missing_data_stops_with_the_expected_paths(
    build: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path / "elsewhere"))
    with pytest.raises(SystemExit, match=r"5k_items_curated\.csv"):
        build.main(["--root", str(tmp_path)])


def test_builds_everything_in_order_with_paths_from_root(
    build: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    _patch(build, monkeypatch, recorder)
    assert build.main([]) == 0
    commands = [c[0] if c[0] != "run" else f"run {c[2]}" for c in recorder.calls]
    assert commands == ["index", "images", "run bm25", "run dense_a", "pool"]
    images = recorder.calls[1]
    assert f"--items={root / 'data' / '5k_items_curated.csv'}" in images
    assert f"--out={root / 'artifacts' / 'images'}" in images
    assert recorder.calls[-1] == ["pool", "--systems", "bm25", "dense_a"]


def test_existing_outputs_are_skipped_unless_forced(
    build: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    _patch(build, monkeypatch, recorder)
    build.main(["--skip-images"])
    paths.items_parquet().parent.mkdir(parents=True, exist_ok=True)
    paths.items_parquet().write_text("", encoding="utf-8")
    recorder.calls.clear()
    monkeypatch.setattr(build, "status_lines", lambda report: [])
    build.main(["--skip-images"])
    assert recorder.calls == []
    build.main(["--skip-images", "--force", "--systems", "bm25"])
    assert [c[0] for c in recorder.calls] == ["index", "run", "pool"]


def test_missing_key_skips_a_system_and_failures_set_exit_code(
    build: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    recorder = Recorder({"run dense_a": openai.OpenAIError("no key"), "index": SystemExit("bad")})
    _patch(build, monkeypatch, recorder)
    assert build.main(["--skip-images"]) == 1
    out = capsys.readouterr().out
    assert "skipped  run dense_a (needs OPENAI_API_KEY" in out
    assert "FAILED   index: bad" in out
    assert recorder.calls[-1] == ["pool", "--systems", "bm25"]


def test_frozen_evaluation_keeps_the_pool_and_qrels_trigger_eval(
    build: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder()
    _patch(build, monkeypatch, recorder)
    paths.artifacts_dir().mkdir(exist_ok=True)
    paths.eval_freeze_json().write_text("{}", encoding="utf-8")
    paths.qrels_json().write_text("{}", encoding="utf-8")
    build.main(["--skip-images"])
    assert [c[0] for c in recorder.calls] == ["index", "run", "run", "eval"]


def test_unknown_system_is_rejected(
    build: ModuleType, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(build, monkeypatch, Recorder())
    with pytest.raises(SystemExit, match="unknown systems"):
        build.main(["--systems", "nope"])
