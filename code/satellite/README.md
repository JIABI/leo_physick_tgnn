# Satellite platform

This package contains the LEO allocation simulator, dynamic candidate graph,
Snapshot and Intensity--Flow controller interfaces, MLP/KAN/PhysiCK models,
training and action-coupled evaluation paths, and the complete v8 satellite
study planner.

The active registry contains 13 studies: `FCT-CL`, `DEC-K`, `STRS-K`,
`LONG-HORIZON-300S`, `FCT-DWELL`, `EPH-REPLAY`, `ORACLE-LADDER`, `ABLATIONS`,
`ROLLOUT-AWARE`, `SCORE-WEIGHT-SENSITIVITY`,
`SNAPSHOT-WEIGHT-SELECTION`, `LEARNED-CONTROLS` and `CLASSICAL-CONTROLS`.
S4 and Mamba2 appear only as exploratory temporal ablations.

Install and inspect the registry:

```bash
python -m pip install -e ".[paper]"
cfs-satellite inventory
```

Create a study plan from the platform protocol and a completed run manifest:

```bash
cfs-satellite plan \
  --config configs/paper_protocol.yaml \
  --studies configs/studies.yaml \
  --author /path/to/paper_artifacts.yaml \
  --select all \
  --out /path/to/satellite_plan
```

The plan records five independent seed bundles and 30 paired held-out episodes
per run. It also records the 3,000-step long-horizon study, the 5/10/20/40 dwell
grid, teacher forcing, scheduled sampling, H=5/10/20 rollout objectives,
score-weight sensitivity, public ephemeris replay and the learned and classical
controls.

See `PAPER_TO_CODE_MAP.md` for module-level traceability.

