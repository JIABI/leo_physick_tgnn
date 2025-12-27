from __future__ import annotations
from typing import List, Dict, Any

def collate_episode(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Recommended: batch_size=1 for dynamic temporal graphs.
    assert len(batch) == 1, "Use batch_size=1 for dynamic graphs in this baseline implementation."
    return batch[0]
