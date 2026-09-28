import subprocess
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import parse_qs

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def shots(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delitem(sys.modules, "screenshots", raising=False)
    import screenshots

    return screenshots


def test_view_urls_deep_link_query_and_systems(shots: ModuleType) -> None:
    urls = shots.view_urls("http://127.0.0.1:9", "Comida para piquenique no parque")

    assert set(urls) == set(shots.VIEW_HEIGHTS)
    view, _, query = urls["search"].partition("#")[2].partition("?")
    assert view == "search"
    assert parse_qs(query) == {"q": ["Comida para piquenique no parque"], "system": ["hybrid"]}
    compare = parse_qs(urls["compare"].partition("?")[2])
    assert compare["left"] == ["dense_pointwise"]
    assert compare["right"] == ["hybrid"]
    assert urls["about"] == "http://127.0.0.1:9/#about"


def test_find_chrome_prefers_explicit_binary(
    shots: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shots.shutil, "which", lambda name: f"/opt/{name}")

    assert shots.find_chrome("my-chrome") == "/opt/my-chrome"
    assert shots.find_chrome() == f"/opt/{shots.CHROME_NAMES[0]}"


def test_find_chrome_exits_when_none_found(
    shots: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shots.shutil, "which", lambda name: None)

    with pytest.raises(SystemExit, match="pass --chrome"):
        shots.find_chrome()


def test_shoot_reports_chrome_failure(
    shots: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(shots.subprocess, "run", lambda *a, **k: failed)

    with pytest.raises(SystemExit, match="boom"):
        shots.shoot("chrome", "http://x/#about", tmp_path / "a.png", (10, 10), 1, tmp_path)


def test_shoot_requires_the_png(
    shots: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ok = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    monkeypatch.setattr(shots.subprocess, "run", lambda *a, **k: ok)

    with pytest.raises(SystemExit, match="Chrome failed"):
        shots.shoot("chrome", "http://x/#about", tmp_path / "a.png", (10, 10), 1, tmp_path)
