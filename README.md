# leo_physick_tgn

Physics-guided Temporal Graph Network (TGN) prototype with three plug-and-play
message functions:

- `mlp`: multilayer-perceptron baseline
- `kan`: spline-based KAN message
- `physick`: legacy analytical kernel bank whose default dot mixer uses
  signed-L1-projected coefficients

The repository now has three deliberately separate paths:

- the legacy frozen-trajectory prototype and its existing data/train/eval
  scripts; and
- a paper-protocol path with separate Intensity--Flow and Snapshot decision
  interfaces, fixed-rank controllers, optional hard feasibility masks, and
  action-coupled transitions; and
- an isolated UAV/shared-service data, model, training and paired-evaluation
  path for the manuscript's second platform.

The second path now has its own typed generator, trainer, checkpoint writer,
paired action-coupled evaluator, baseline registry, controller/calibration
utilities, and one-command shell pipeline. It does not silently route through
the legacy rollout scripts. See [`PAPER_PROTOCOL.md`](PAPER_PROTOCOL.md) for the
formal entry points and [`PAPER_PARAMETER_AUDIT.md`](PAPER_PARAMETER_AUDIT.md)
for reported versus estimated parameters.

## Paper-protocol core

The default executable method matrix is:

```bash
METHODS=tgn_mlp,tgn_kan,tgn_physick,snapshot_mlp,snapshot_physick,snapshot_ltt_r,snapshot_da_gwm,da_gwm,s4,mamba2,conformer,big_mlp,edge_attn,temp_trans \
  bash scripts/paper_protocol.sh
```

This command installs selected official optional backends, generates the typed
paper dataset, trains each selected method, automatically writes best/last
checkpoints, and invokes paired action-coupled evaluation. Individual stage
commands and output contracts are documented in `PAPER_PROTOCOL.md`.
When `snapshot_physick` is selected, the default full protocol also creates
the scheduled-sampling condition and action-coupled rollout-loss conditions at
`H={5,10,20}`. Set `RUN_SNAPSHOT_SCHEDULED_SAMPLING=0` or
`RUN_SNAPSHOT_ROLLOUT_LADDER=0` to skip them.

The unified SI runtime profiler requires a formal test dataset and a strictly
compatible checkpoint; it has no synthetic-input fallback and never constructs
an optimizer. For example:

```bash
python scripts/profile_runtime.py \
  --platform ntn --cfg configs/paper_protocol.yaml --method tgn_physick \
  --ckpt artifacts/paper_runs/tgn_physick/checkpoint.pt \
  --data artifacts/data/ntn_paper_schema_v1.pt \
  --device cuda --warmup 20 \
  --json-out artifacts/paper_runs/tgn_physick/runtime_profile.json \
  --csv-out artifacts/paper_runs/tgn_physick/runtime_profile.csv
```

Use `--platform snapshot` with its method-specific Snapshot dataset, or
`--platform uav --run-seed <seed>` with the matching independent UAV checkpoint.
The default and mandatory main row is `full_decision_epoch`; add
`--include-model-predict-step` only for the optional fixed-graph diagnostic.
The default repeat count is the formal evaluation horizon (600 for NTN and
Snapshot, 120 for UAV), independent of the shorter training shard horizon.
The JSON retains raw latency samples and provenance; the CSV includes Params M
and peak memory GiB. CUDA timing synchronizes every sample. The artifact marks
`si_runtime_table_comparable=true` only for a batch-one, complete 600-step
full-decision run from `t=0` on an NVIDIA A100. CPU results and CPU peak memory
are diagnostic; `ru_maxrss` is a process-lifetime high-water mark whose
observed stage delta can be zero after an earlier allocation peak. Set
`RUN_RUNTIME_PROFILE=1` on `paper_protocol.sh` to enable the optional NTN and
Snapshot profiling stage after evaluation.

Here `s4`, `mamba2`, and `conformer` are temporal-mixer ablations of the same
two-layer, width-128 Paper PhysiCK graph and Intensity--Flow readout. They
replace the GRU update only; they are not independent width-256 graph models.
LTT-R and DA-GWM remain separate controlled baselines.
BigMLP and EdgeAttn retain the same two graph-message layers, 128-dimensional
memory/message/readout, GRU updates and Intensity--Flow head as the Paper TGN;
they replace only the message operator with, respectively, a widened
two-linear-layer MLP or GAT-style attention. TempTrans retains the standard MLP
graph operators and readout and replaces only GRU temporal mixing with a
per-node Transformer. Their unreported internal widths/depths are explicit
estimates constrained to the Supplementary runtime table's rounded parameter
classes (2.1M, 2.1M and 2.4M), rather than to the parameter count of the current
reconstructed PhysiCK implementation. Exact counts and manuscript provenance
are recorded in the parameter audit and frozen checkpoint config.

`snapshot_mlp` and `snapshot_physick` are the two Snapshot-side cells in the
interface-by-operator factorial. They use formal Snapshot-oracle trajectories
stored separately from the Intensity--Flow dataset. `snapshot_ltt_r` and
`snapshot_da_gwm` are also registered in the default full method matrix;
their Snapshot score weights are exposed estimates unless a recovered frozen
configuration is supplied. Snapshot data, checkpoints and paired traces never
reuse the Intensity--Flow bundle merely by renaming its fields.

The stable Intensity--Flow Python entry points are:

```python
from pathlib import Path

import yaml

from leo_pg.eval import (
    ConstantPolicyStreamInitializer,
    TGNDescriptorProvider,
    run_closed_loop,
)
from leo_pg.models import build_model
from leo_pg.sim import PaperAlignedLEOEnv

cfg = yaml.safe_load(Path("configs/paper_core.yaml").read_text())
env = PaperAlignedLEOEnv(cfg)
model = build_model(cfg)  # head.type must be "intensity_flow"
provider = TGNDescriptorProvider(model, device="cpu")
initializer = ConstantPolicyStreamInitializer.from_config(cfg)

model_result = run_closed_loop(
    env,
    mode="model",
    descriptor_provider=provider,
    policy_initializer=initializer,
)
oracle_result = run_closed_loop(
    PaperAlignedLEOEnv(cfg),
    mode="oracle",
)
```

`configs/paper_core.yaml` is an executable API example, not an experiment or a
source of paper results. Its explicitly marked `phi`, baseline-hazard, and Cox
scale choices must be replaced by the frozen values used for any formal run.

`run_closed_loop` uses an explicit one-step staging rule. The action at epoch
`t` consumes the policy descriptors already staged for `t`; the model output
computed at `t` is retained for `t+1` and is never used prematurely for the
current action. Persistent edge fields are joined by local `(user, satellite)`
identity. A model rollout must supply a fingerprinted policy-stream initializer
for D0 and for any newly appearing candidate edge. Missing values are never
silently encoded as the semantically meaningful intensity value zero, and they
are never hidden by changing the controller candidate set. The configured
prior in the example is a no-teacher-warm-start API value, not a recovered paper
parameter; a formal run can replace it with a learned initializer.

The environment lifecycle is not Gym-compatible by accident; it is explicit:

```text
reset_control() -> ControlObservation
step_action(ServingAction) -> (next_observation | None, ExecutionResult, done)
```

Actions carry the originating `observation_id`, use local satellite indices,
and use `-1` for abstention. Stale or repeated actions fail. Simulator-side
feasibility and capacity remain authoritative; a rejected request has
`executed_serving=-1` for that epoch. Policy descriptor replacement never
writes into simulator state.

### Feasibility protocols

`paper_protocol.policy.hard_feasibility_mask` is mandatory rather than hidden
behind a default:

- `false`: rank the complete geometry-defined Top-k set, then let the simulator
  reject an infeasible request;
- `true`: remove simulator-infeasible candidates before rank normalization,
  while retaining the same execution-layer gate.

The learned feasibility logit is auxiliary in both cases. It is never accepted
as a controller mask or simulator predicate.

### Oracle substitution modes

Partial modes start from a complete model prediction and replace the named
field with its current simulator oracle value:

| Mode | gamma | intensity | flow |
|---|---:|---:|---:|
| `model` | model | model | model |
| `oracle` | oracle | oracle | oracle |
| `gamma` | oracle | model | model |
| `intensity` | model | oracle | model |
| `flow` | model | model | oracle |
| `intensity_flow` | model | oracle | oracle |

Only the policy-facing copy changes. Candidate geometry, topology, history,
hard constraints, simulator descriptors, and admission state are cloned
unchanged. In partial-oracle modes, the corrected copy is evaluated only by the
current fixed controller; the next-step predictor continues from the staged
model stream, so the intervention does not become a hidden teacher-forcing
input.

### Feature and prediction contracts

The two paper interfaces have separate data, model-input, target, controller,
training and evaluation contracts. Snapshot checkpoints record
`paper_interface=snapshot_v1`, and both pipelines validate their own dataset
kind/feature contract rather than accepting data from the other interface.

#### Intensity--Flow

Paper-core node order is users followed by satellites. `node_x` has 7 columns:
position (3), velocity (3), and the policy-facing satellite flow carrier (user
rows are zero). Candidate edges run from user nodes to satellite nodes and have
7 columns: normalized elevation, log distance, normalized policy gamma,
policy destination flow, `log1p` policy intensity, current-association flag,
and normalized dwell. `candidate_edge_ids` stores local `(user, satellite)`
indices; it is not a NORAD identifier.

Direct model use therefore requires `model.node_in_dim: 7`,
`model.edge_in_dim: 7`, and `head.type: intensity_flow`. The head returns edge
gamma, non-negative edge intensity, satellite flow in `[0,1]`, and a separate
auxiliary feasibility logit. `ControlObservation.as_model_step()` materializes
features from the recursively carried policy copy and never includes `y`.

For training, `leo_pg.paper.dataset` and `leo_pg.paper.training` provide typed
next-step targets and the multi-task objective: edge gamma MSE, edge
`log1p(intensity)` MSE, satellite-flow MSE, and feasibility BCE all use the
specified domains, with edge losses restricted to persistent links. The paper
trainer also supports scheduled sampling, action-coupled multi-step rollout,
validation-selected checkpoints, and DA-GWM's additive decision-aware loss.
Loss weights are explicit because the manuscript materials do not fix a unique
numeric weighting. The legacy `Trainer` remains unchanged.

#### Snapshot

Snapshot uses the same user-then-satellite graph identity and seven-column
widths, but rebuilds the tensors from a Snapshot-only allow-list. Its model
input contains position, velocity, current geometry, current gamma cue, a
current feasibility margin, current executed admitted-count divided by
satellite capacity, association history and dwell. Neither the real-valued
descriptors nor the feature tensors read integrated intensity, `L_t`, `Phi`, or
the EMA update. The typed head and target return gamma, feasibility margin and
satellite admitted load, with edge losses restricted to persistent candidates.

Formal data must be produced by the Snapshot oracle controller through
`paper_snapshot_generate.py`. Converting an Intensity--Flow trajectory is
diagnostic-only because it preserves the Intensity--Flow behaviour actions and
their induced state distribution. Snapshot model/oracle evaluation uses the
same staged rule as the main protocol: a prediction made at `t` is joined onto
persistent candidates at `t+1`, and D0/new edges require an explicit
initializer.

The simulator's boolean feasibility mask remains authoritative for execution
and, when configured, the controller's hard mask. It is not a predicted
Snapshot field and does not reintroduce the Intensity--Flow congestion state.

## Verified end-to-end smoke test

The commands below exercise the complete local path: package installation, data
generation, training, checkpoint loading, evaluation, and rollout export.

```bash
python -m pip install -e ".[test]"
python -m pytest -q

python scripts/gen_data.py \
  --cfg configs/data/synthetic_debug.yaml \
  --out data/synthetic_debug.pt \
  --episodes 9 \
  --split-ratios 0.67,0.11,0.22 \
  --device cpu

python dataset_verify.py data/synthetic_debug.pt

python scripts/train.py \
  --cfg configs/smoke.yaml \
  --message physick

python scripts/eval.py \
  --cfg configs/smoke.yaml \
  --message physick \
  --which last \
  --split test

python scripts/rollout.py \
  --cfg configs/smoke.yaml \
  --message physick \
  --which last \
  --split test \
  --Hs 5,10 \
  --max_eps 2
```

The shell wrappers `gen_dataset.sh`, `train.sh`, and `rollout.sh` run the same
workflow and accept environment-variable overrides.

Expected outputs are `data/synthetic_debug.pt`,
`runs/smoke_physick/last.pt`, and
`runs/smoke_physick/rollout_metrics.json`. This workflow was verified on macOS
14.7.6 (Apple silicon), Python 3.12.0, PyTorch 2.5.1, and NumPy 2.1.3. The
declared package floor remains Python 3.10 and PyTorch 2.1.

## Repository scope

The implemented NTN path includes separate formal Intensity--Flow and Snapshot
data generation, training, checkpoints and action-coupled evaluation; all
method names in `PAPER_METHODS`; classical controllers; calibration; the three
runtime-table baselines; and shield/shrink-jump audits. The Snapshot
factorial cells are `snapshot_mlp` and `snapshot_physick`, while
`snapshot_ltt_r` and `snapshot_da_gwm` are included in the full default matrix.

The repository still cannot recover the reported
numbers because the original data, checkpoints, exact unreported parameters
and run manifests were lost. The UAV graph dataset, recurrent descriptor model,
trainer, per-run checkpoints, substitution evaluator and metrics are present;
its full operator/interface factorial, stress-grid runner and figure pipeline
remain future work. The manuscript figure renderer is also absent.
`PAPER_IMPLEMENTATION_GAP.md` records the exact coverage.

The evaluation and rollout commands use the held-out episode split. The rollout
command is a teacher-forced diagnostic on a frozen, pre-generated
trajectory. Its adjacent A-B-A and assignment-failure fractions are software
diagnostics, not the manuscript's closed-loop ping-pong and handover-failure
endpoints.

## Data schema (torch .pt)

A saved dataset is a dict:
```python
{
  "dataset_schema_version": 2,
  "episodes": [
    {
      "steps": [
        {"t": int,
         "node_x": FloatTensor[N, F_node],
         "edge_index": LongTensor[2, E],
         "edge_z": FloatTensor[E, F_edge],
         "edge_type": LongTensor[E],
         "y": FloatTensor[N, F_out],   # supervision target
         "meta": {
           "K_users": int,
           "S_sats": int,
           "sat_load_pre": FloatTensor[S],
           "allocation_user_order": LongTensor[K],
           "allocator_protocol": {...},
           "serving_sat": LongTensor[K],
           "ho_fail": BoolTensor[K]
         }
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

Loaders reject missing or incompatible schema versions so that older bundles
cannot be silently evaluated under the repaired allocator protocol.

This prototype supports `batch_size=1` for dynamic graphs. Disjoint-union
batching is not implemented.



## Real ephemeris (SGP4 via Skyfield)

```bash
pip install skyfield sgp4
```

See `src/leo_pg/sim/ephemeris.py` for `SkyfieldTLEEphemeris` and `HybridUserSatEphemeris`.

## Data verification

```bash
python dataset_verify.py data/synthetic_debug.pt
```

The verifier checks the saved schema, finite tensor values, split sizes, and
representative graph dimensions.
