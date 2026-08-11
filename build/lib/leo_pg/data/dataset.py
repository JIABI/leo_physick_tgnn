from __future__ import annotations
from torch.utils.data import Dataset
from leo_pg.data.io import load_bundle

class TemporalEpisodeDataset(Dataset):
    def __init__(self, path: str, split: str = "train"):
        payload = load_bundle(path)
        if "episodes" not in payload:
            raise KeyError("Expected dataset bundle with key 'episodes'")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train|val|test")

        # A bundle may contain explicit, mutually exclusive splits. Legacy
        # unsplit bundles are train-only; returning them for val/test would leak
        # the training episodes into evaluation.
        if split in payload:
            self.episodes = payload[split]
        elif split == "train":
            self.episodes = payload["episodes"]
        else:
            raise ValueError(
                f"Dataset {path!r} has no explicit {split!r} split. "
                "Regenerate it with scripts/gen_data.py or evaluate on split='train'."
            )

        if len(self.episodes) == 0:
            raise ValueError(f"Dataset split {split!r} in {path!r} is empty")
        self.meta = payload.get("meta", {})

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.episodes[idx]
