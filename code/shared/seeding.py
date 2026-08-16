"""Deterministic and auditable seed derivation."""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np


def derive_seed(root_seed: int, namespace: str, index: int = 0) -> int:
    """Derive a stable 32-bit seed without relying on Python's salted hash."""

    payload = f"controller-facing-state-v1|{int(root_seed)}|{namespace}|{int(index)}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big")


@dataclass(frozen=True)
class SeedBundle:
    training_seed: int
    model_init_seed: int
    optimizer_seed: int
    data_order_seed: int
    environment_seed: int

    @classmethod
    def from_training_seed(cls, training_seed: int) -> "SeedBundle":
        return cls(
            training_seed=int(training_seed),
            model_init_seed=derive_seed(training_seed, "model_init"),
            optimizer_seed=derive_seed(training_seed, "optimizer"),
            data_order_seed=derive_seed(training_seed, "data_order"),
            environment_seed=derive_seed(training_seed, "environment"),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "training_seed": self.training_seed,
            "model_init_seed": self.model_init_seed,
            "optimizer_seed": self.optimizer_seed,
            "data_order_seed": self.data_order_seed,
            "environment_seed": self.environment_seed,
        }


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch when installed.

    This function does not claim bitwise equivalence across hardware or library
    versions. The returned record says which backends were actually seeded.
    """

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    status: dict[str, Any] = {"python": True, "numpy": True, "torch": False}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
        status["torch"] = True
        status["torch_deterministic_algorithms"] = bool(deterministic_torch)
    except ImportError:
        pass
    return status


def numpy_generator(root_seed: int, namespace: str, index: int = 0) -> np.random.Generator:
    return np.random.default_rng(derive_seed(root_seed, namespace, index))


def episode_seed_panel(root_seed: int, count: int, *, panel_id: str) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [derive_seed(root_seed, f"episode-panel:{panel_id}", i) for i in range(count)]
