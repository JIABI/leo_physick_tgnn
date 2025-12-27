# leo_physick_tgn

Physics-Guided Temporal Graph Network (TGN) with plug-and-play message functions:
- `mlp` baseline
- `kan` spline-based KAN message
- `physick` (kernel bank + KAN coefficient mixing) message

## Quickstart

```bash
# 1) create synthetic debug data
python scripts/gen_data.py --cfg configs/data/synthetic_debug.yaml --out data/sim.pt

# 2) train (choose message_type in config or override via CLI)
python scripts/train.py --cfg configs/default.yaml --message physick

# 3) evaluate one-step and rollout
python scripts/eval.py --cfg configs/default.yaml
python scripts/rollout.py --cfg configs/default.yaml --H 30
```

## Data schema (torch .pt)

A saved dataset is a dict:
```python
{
  "episodes": [
    {
      "steps": [
        {"t": int,
         "node_x": FloatTensor[N, F_node],
         "edge_index": LongTensor[2, E],
         "edge_z": FloatTensor[E, F_edge],
         "edge_type": LongTensor[E],   # optional (0 if unused)
         "y": FloatTensor[N, F_out],   # supervision target
        },
        ...
      ],
      "meta": {...}
    },
    ...
  ],
  "meta": {...}
}
```

This repo supports batch_size=1 for dynamic graphs by default (recommended). Disjoint union batching scaffold exists but is off by default.



## Real ephemeris (SGP4 via Skyfield)

```bash
pip install skyfield sgp4
```

See `src/leo_pg/sim/ephemeris.py` for `SkyfieldTLEEphemeris` and `HybridUserSatEphemeris`.
