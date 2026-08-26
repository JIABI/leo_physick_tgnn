"""Frozen regional-visibility source used by the balanced satellite factorial."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .randomness import keyed_generator, keyed_standard_normal


@dataclass(frozen=True)
class FrozenVisibilityConfig:
    mean: float = 47.0
    standard_deviation: float = 3.2
    minimum: int = 38
    maximum: int = 58

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("visibility mean must be finite")
        if not math.isfinite(self.standard_deviation) or self.standard_deviation <= 0:
            raise ValueError("visibility standard deviation must be finite and positive")
        if self.minimum <= 0 or self.maximum < self.minimum:
            raise ValueError("visibility bounds must be positive and ordered")


def sample_frozen_visibility_count(
    *,
    run_seed: int,
    episode_index: int,
    config: FrozenVisibilityConfig = FrozenVisibilityConfig(),
    stress_adjustment: int = 0,
) -> int:
    """Draw once per matched run--episode, then reuse across factorial cells."""

    if run_seed < 0 or episode_index < 0:
        raise ValueError("run_seed and episode_index must be non-negative")
    if isinstance(stress_adjustment, bool) or int(stress_adjustment) != stress_adjustment:
        raise TypeError("stress_adjustment must be an integer")
    z = float(
        keyed_standard_normal(
            (1,), run_seed, episode_index, "factorial-regional-visibility"
        )[0].item()
    )
    sampled = int(round(config.mean + config.standard_deviation * z))
    sampled += int(stress_adjustment)
    return max(config.minimum, min(config.maximum, sampled))


def select_frozen_regional_satellites(
    *,
    satellite_count: int,
    visible_count: int,
    run_seed: int,
    episode_index: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Select the shared regional satellite identities for one run--episode.

    The returned identities are sampled once, independently of user geometry,
    and are therefore shared by every user and every matched factorial cell.
    Per-user elevation masking and Top-k thinning happen only after this set has
    been fixed.
    """

    if isinstance(satellite_count, bool) or int(satellite_count) != satellite_count:
        raise TypeError("satellite_count must be an integer")
    if isinstance(visible_count, bool) or int(visible_count) != visible_count:
        raise TypeError("visible_count must be an integer")
    satellite_count = int(satellite_count)
    visible_count = int(visible_count)
    if satellite_count <= 0:
        raise ValueError("satellite_count must be positive")
    if not 0 < visible_count <= satellite_count:
        raise ValueError("visible_count must lie in [1, satellite_count]")
    if run_seed < 0 or episode_index < 0:
        raise ValueError("run_seed and episode_index must be non-negative")
    selected = torch.randperm(
        satellite_count,
        generator=keyed_generator(
            run_seed,
            episode_index,
            "factorial-regional-satellite-subset",
        ),
        device="cpu",
    )[:visible_count]
    # Sorting gives the set a canonical representation. Candidate ranking still
    # uses elevation first and global satellite identifier for exact ties.
    return torch.sort(selected).values.to(device=device)


__all__ = [
    "FrozenVisibilityConfig",
    "sample_frozen_visibility_count",
    "select_frozen_regional_satellites",
]
