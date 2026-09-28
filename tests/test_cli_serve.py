import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from foodsearch import cli
from foodsearch.api.backend import NotFoundError


def test_serve_binds_localhost_on_the_given_port(monkeypatch: pytest.MonkeyPatch) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "create_app", lambda config: served.setdefault("config", config))
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: served.update(kwargs))
    monkeypatch.setattr(cli, "port_is_free", lambda port: True)
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


@pytest.fixture
def busy_port() -> Iterator[int]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]


def test_port_is_free_detects_a_listening_socket(busy_port: int) -> None:
    assert not cli.port_is_free(busy_port)
    assert cli.next_free_port(busy_port) not in (None, busy_port)


def serve_on_busy_port(
    monkeypatch: pytest.MonkeyPatch, busy_port: int, answer: str, tty: bool = True
) -> dict[str, Any]:
    served: dict[str, Any] = {}
    monkeypatch.setattr(cli, "create_app", lambda config: object())
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: served.update(kwargs))
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: tty)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    cli.main(["serve", "--port", str(busy_port)])
    return served


@pytest.mark.parametrize("answer", ["", "y", "YES"])
def test_serve_moves_to_the_next_free_port_when_confirmed(
    monkeypatch: pytest.MonkeyPatch, busy_port: int, answer: str
) -> None:
    served = serve_on_busy_port(monkeypatch, busy_port, answer)
    assert served["port"] != busy_port
    assert cli.port_is_free(served["port"])


@pytest.mark.parametrize(("answer", "tty"), [("n", True), ("", False)])
def test_serve_exits_when_the_port_is_busy_and_not_confirmed(
    monkeypatch: pytest.MonkeyPatch, busy_port: int, answer: str, tty: bool
) -> None:
    with pytest.raises(SystemExit, match="in use"):
        serve_on_busy_port(monkeypatch, busy_port, answer, tty)
