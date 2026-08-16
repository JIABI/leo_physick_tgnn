from __future__ import annotations
from typing import Any, Dict

from torch.utils.data import Dataset
from leo_pg.data.io import load_bundle
from leo_pg.data.schema import DATASET_SCHEMA_VERSION

class TemporalEpisodeDataset(Dataset):
    def __init__(self, path: str, split: str = "train"):
        payload = load_bundle(path)
        schema_version = payload.get("dataset_schema_version")
        if schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Dataset schema version {schema_version!r} is unsupported; expected "
                f"{DATASET_SCHEMA_VERSION}. Regenerate the bundle with scripts/gen_data.py."
            )
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
        self.schema_version = int(schema_version)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.episodes[idx]
