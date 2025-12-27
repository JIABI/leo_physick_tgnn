from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from leo_pg.data.io import load_bundle

class TemporalEpisodeDataset(Dataset):
    def __init__(self, path: str, split: str = "train"):
        payload = load_bundle(path)
        assert "episodes" in payload, "Expected payload with key 'episodes'"
        assert split in ["train", "val", "test"], "split must be train|val|test"
        # Payload may already be split; if not, assume all episodes belong to 'train'.
        self.episodes = payload.get(split, None)
        if self.episodes is None:
            self.episodes = payload["episodes"]
        self.meta = payload.get("meta", {})

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.episodes[idx]
