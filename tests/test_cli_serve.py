from pathlib import Path
from typing import Any

import pytest

from foodsearch import cli
from foodsearch.api.backend import NotFoundError


def test_serve_binds_localhost_on_the_given_port(monkeypatch: pytest.MonkeyPatch) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "create_app", lambda config: served.setdefault("config", config))
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: served.update(kwargs))
    cli.main(["serve", "--port", "8123", "--config", "custom.yaml"])
    assert served == {"config": Path("custom.yaml"), "host": "127.0.0.1", "port": 8123}


def test_serve_exits_with_a_hint_when_artifacts_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(config: Path) -> None:
        raise NotFoundError("items.parquet not found; run `python scripts/build_artifacts.py`")

    monkeypatch.setattr(cli, "create_app", missing)
    with pytest.raises(SystemExit, match="build_artifacts"):
        cli.main(["serve"])
