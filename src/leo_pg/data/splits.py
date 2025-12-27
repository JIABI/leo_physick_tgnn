from __future__ import annotations
from typing import Dict, List, Any, Tuple

def split_episodes(episodes: List[Any], ratios=(0.8, 0.1, 0.1), seed: int = 7) -> Dict[str, List[Any]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(episodes)
    idx = list(range(n))
    import random
    random.Random(seed).shuffle(idx)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    tr = [episodes[i] for i in idx[:n_tr]]
    va = [episodes[i] for i in idx[n_tr:n_tr+n_va]]
    te = [episodes[i] for i in idx[n_tr+n_va:]]
    return {"train": tr, "val": va, "test": te}
