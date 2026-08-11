# UAV/shared-service platform core

This package is the isolated simulator core for the manuscript's second
platform.  It does not reuse the NTN transition, controller, data schema or
training pipeline.

Implemented protocol surface:

- bounded-turn two-dimensional UAV motion and finite energy;
- reachable Top-3 service-station candidates with deterministic tie-breaking;
- six stations, two active service slots and FIFO queues of capacity eight;
- service completion, queue admission/backfill and reassociation execution;
- policy-facing `eta`, feasible-start intensity and slow station-flow fields;
- separate simulator-authoritative and policy descriptor copies;
- action-coupled `reset_control() -> observe() -> step_action(action)` timing;
- keyed exogenous randomness and protocol-bound snapshots/forks.
- typed UAV/station/candidate graph and persistent-edge next-step targets;
- a target-free recurrent graph model with eta, intensity, feasibility and
  station-flow heads;
- one-step, scheduled-sampling and action-coupled multi-step training code;
- six-condition model/oracle/partial-oracle substitution evaluation; and
- episode endpoints plus hierarchical run/episode paired bootstrap summaries.

The values stated numerically in the manuscript are immutable fields of
`ManuscriptExactParameters`.  Unreported region, motion, energy and service
constants live only in `EstimatedDynamicsParameters`; the protocol manifest
labels every one as `explicit_estimate`.

The repository-level paper configuration is the preferred construction path:

```python
from leo_pg.uav_shared import build_uav_environment

env = build_uav_environment("configs/paper_protocol.yaml")
observation = env.reset_control()
```

For a deliberately small standalone use, the same environment can also be
constructed from the typed protocol directly:

```python
from leo_pg.uav_shared import (
    ReassociationAction,
    UAVSharedProtocol,
    UAVSharedServiceEnv,
)

protocol = UAVSharedProtocol(
    episode_seed=17,
    density_multiplier=1,
    capacity_compression=1.0,
)
env = UAVSharedServiceEnv(protocol)
observation = env.reset_control()

# A real controller supplies one local station id per UAV; -1 abstains.
action = ReassociationAction(
    observation_id=observation.observation_id,
    requested_station=observation.current_station.clone(),
)
next_observation, execution, done = env.step_action(action)
```

The command-line pipeline is split into three explicit stages.  Formal data use
five independent run seeds, training writes one `best.pt` and `last.pt` pair per
run, and evaluation requires the matching checkpoint for every run:

```bash
python scripts/uav_generate.py \
  --cfg configs/paper_protocol.yaml \
  --out artifacts/uav/uav_dataset.pt

python scripts/uav_train.py \
  --cfg configs/paper_protocol.yaml \
  --data artifacts/uav/uav_dataset.pt \
  --out artifacts/uav/checkpoints \
  --all-runs

python scripts/uav_evaluate.py \
  --cfg configs/paper_protocol.yaml \
  --data artifacts/uav/uav_dataset.pt \
  --ckpt artifacts/uav/checkpoints \
  --out artifacts/uav/evaluation
```

These commands define the executable protocol; running them is deliberately
separate from importing the package.
