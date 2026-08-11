from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence


def _split_counts(n: int, ratios: Sequence[float]) -> List[int]:
    if len(ratios) != 3:
        raise ValueError("ratios must contain exactly three values: train, val, test")
    if any(float(ratio) < 0.0 for ratio in ratios):
        raise ValueError("split ratios must be non-negative")
    total = float(sum(ratios))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    raw = [float(ratio) * n for ratio in ratios]
    counts = [int(math.floor(value)) for value in raw]
    remainder = n - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    # If the dataset is large enough, do not silently create an empty requested
    # validation or test split. Move one item from the largest donor split.
    positive = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if n >= len(positive):
        for index in positive:
            if counts[index] == 0:
                donors = [candidate for candidate in positive if counts[candidate] > 1]
                if donors:
                    donor = max(donors, key=lambda candidate: counts[candidate])
                    counts[donor] -= 1
                    counts[index] += 1
    return counts


def split_episodes(
    episodes: List[Any],
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 7,
) -> Dict[str, List[Any]]:
    """Create deterministic, mutually exclusive episode splits."""
    indices = list(range(len(episodes)))
    random.Random(int(seed)).shuffle(indices)
    n_train, n_val, n_test = _split_counts(len(indices), ratios)
    train_end = n_train
    val_end = n_train + n_val
    return {
        "train": [episodes[index] for index in indices[:train_end]],
        "val": [episodes[index] for index in indices[train_end:val_end]],
        "test": [episodes[index] for index in indices[val_end:val_end + n_test]],
    }
