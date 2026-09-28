import pytest

from foodsearch import device
from foodsearch.device import DEVICE_ENV_VAR, resolve_device


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    available = {"cuda": False, "mps": True, "cpu": True}
    monkeypatch.setattr(device, "is_available", lambda d: available[d])
    assert resolve_device() == "mps"
    available["cuda"] = True
    assert resolve_device() == "cuda"


def test_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "is_available", lambda d: d == "cpu")
    assert resolve_device() == "cpu"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEVICE_ENV_VAR, "CPU")
    assert resolve_device() == "cpu"


def test_explicit_argument_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEVICE_ENV_VAR, "cuda")
    assert resolve_device("cpu") == "cpu"


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_unavailable_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "is_available", lambda d: d == "cpu")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
