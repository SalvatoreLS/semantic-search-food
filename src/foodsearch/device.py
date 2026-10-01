import os
from typing import Literal

Device = Literal["cuda", "mps", "cpu"]

DEVICE_ENV_VAR = "FOODSEARCH_DEVICE"
_DEVICES: tuple[Device, ...] = ("cuda", "mps", "cpu")


def is_available(device: Device) -> bool:
    import torch

    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


def resolve_device(preferred: str | None = None) -> Device:
    requested = (preferred or os.environ.get(DEVICE_ENV_VAR) or "").strip().lower()
    if requested:
        if requested not in _DEVICES:
            raise ValueError(f"Unknown device {requested!r}; expected one of {_DEVICES}")
        if not is_available(requested):
            raise RuntimeError(f"Device {requested!r} was requested but is not available")
        return requested
    return next(d for d in _DEVICES if is_available(d))
