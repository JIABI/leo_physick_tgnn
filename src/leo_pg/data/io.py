from __future__ import annotations
from typing import Any, Dict
import torch

def save_bundle(path: str, payload: Dict[str, Any]) -> None:
    torch.save(payload, path)

def load_bundle(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
