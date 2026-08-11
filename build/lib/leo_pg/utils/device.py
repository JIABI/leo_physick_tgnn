from __future__ import annotations
import warnings

import torch


def get_device(device_str: str, *, strict: bool = False) -> torch.device:
    requested = str(device_str).lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device_str!r}")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        message = "CUDA was requested but is unavailable"
        if strict:
            raise RuntimeError(message)
        warnings.warn(message + "; falling back to CPU", RuntimeWarning, stacklevel=2)
    return torch.device("cpu")
