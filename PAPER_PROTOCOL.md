# NTN paper protocol code

This repository exposes a separate paper path. It does not reuse the legacy
frozen-trajectory generator, scalar-load trainer, or post-hoc rollout
diagnostics.

## One-command pipeline

```bash
METHODS=tgn_mlp,tgn_kan,tgn_physick,snapshot_mlp,snapshot_physick,snapshot_ltt_r,snapshot_da_gwm,da_gwm,s4,mamba2,conformer,big_mlp,edge_attn,temp_trans \
  bash scripts/paper_protocol.sh
```

When executed, the script performs these stages in order:

1. installs the selected official optional dependencies;
2. generates typed action-coupled trajectories with train/validation/test
   manifests, using separate formal datasets for Intensity--Flow and each
   selected Snapshot behaviour policy;
3. trains every selected method and writes an automatically selected best
   checkpoint plus `last.pt`;
4. loads the frozen checkpoint and runs paired action-coupled evaluation:
   model/oracle plus partial gamma/intensity/flow substitutions for
   Intensity--Flow, and model/oracle for Snapshot;
5. writes complete trace bundles and JSON manifests.

Selecting `snapshot_physick` additionally runs the scheduled-sampling control
and the `H={5,10,20}` action-coupled rollout-loss ladder. These can be disabled
with `RUN_SNAPSHOT_SCHEDULED_SAMPLING=0` and
`RUN_SNAPSHOT_ROLLOUT_LADDER=0`; set `SNAPSHOT_ROLLOUT_HORIZONS` to change the
explicit horizon list.

These commands describe the executable protocol; their presence is not evidence
that the lost publication experiments or reported numbers were rerun.

Common overrides are environment variables:

```bash
CONFIG=/path/to/protocol.yaml
DATASET=/path/to/paper_dataset.pt
RUN_ROOT=/path/to/runs
METHODS=tgn_physick,ltt_r
SNAPSHOT_DATA_ROOT=/path/to/snapshot_datasets
REUSE_DATASET=1
REUSE_CHECKPOINTS=1
TRAIN_DEVICE=cuda
EVAL_DEVICE=cuda
# Set to 0 only when intentionally skipping the full classical-controller grid.
CONTROLLER_SWEEP=1
bash scripts/paper_protocol.sh
```

`GENERATE_EXTRA_ARGS`, `TRAIN_EXTRA_ARGS`, and `EVAL_EXTRA_ARGS` append
advanced command-line options. Optional controller, metric, shield, and audit
branches are controlled by the corresponding `EVAL_*` variables documented in
the shell script.

## Individual stages

```bash
python scripts/paper_generate.py \
  --cfg configs/paper_protocol.yaml \
  --out artifacts/data/ntn_paper_schema_v1.pt \
  --storage-layout episode_shards \
  --episodes 2230 \
  --split-counts 2000,200,30

python scripts/paper_train.py \
  --cfg configs/paper_protocol.yaml \
  --data artifacts/data/ntn_paper_schema_v1.pt \
  --method tgn_physick \
  --out artifacts/checkpoints/tgn_physick/checkpoint.pt \
  --device cuda

python scripts/paper_evaluate.py \
  --cfg configs/paper_protocol.yaml \
  --ckpt artifacts/checkpoints/tgn_physick/checkpoint.pt \
  --method tgn_physick \
  --out artifacts/evaluation/tgn_physick_paired_trace.pt \
  --device cuda
```

Snapshot methods use their isolated entry points rather than the three
Intensity--Flow commands above:

```bash
python scripts/paper_snapshot_generate.py \
  --cfg configs/paper_protocol.yaml \
  --out artifacts/data/snapshot/snapshot_mlp_snapshot_schema_v1.pt \
  --method snapshot_mlp \
  --episodes 2230 \
  --split-counts 2000,200,30

python scripts/paper_snapshot_train.py \
  --cfg configs/paper_protocol.yaml \
  --data artifacts/data/snapshot/snapshot_mlp_snapshot_schema_v1.pt \
  --method snapshot_mlp \
  --out artifacts/checkpoints/snapshot_mlp/checkpoint.pt \
  --device cuda

python scripts/paper_snapshot_evaluate.py \
  --cfg configs/paper_protocol.yaml \
  --ckpt artifacts/checkpoints/snapshot_mlp/checkpoint.pt \
  --method snapshot_mlp \
  --out artifacts/evaluation/snapshot_mlp_paired_trace.pt \
  --device cuda
```

Formal generation writes `ntn_paper_schema_v1.pt` as a lightweight index and
stores one episode at a time below the adjacent
`ntn_paper_schema_v1_episodes/` directory. The index records each shard's
relative path, byte size, SHA-256 digest, split, episode identity, seed and
protocol fingerprint. Upload or move the index and its adjacent shard directory
together; relative paths make the pair relocatable. Training loads the index
with `weights_only=True`, then verifies and loads only the requested episode.
`--storage-layout monolithic` remains available for deliberately small fixtures.

Snapshot generation writes the same index-plus-shard layout under
`SNAPSHOT_DATA_ROOT`, but its dataset kind, feature contract and target contract
are distinct. It advances `PaperAlignedLEOEnv` with the selected Snapshot oracle
controller. The shell creates a separate Snapshot dataset per method because
the SI-selected MLP and PhysiCK score weights differ. An existing
Intensity--Flow episode can be converted only through the explicitly diagnostic
adapter; that conversion is rejected as formal factorial training data.

Each Intensity--Flow episode stores the controller input and simulator-authoritative channel
separately, the requested and executed action, failure reason, flow before and
after execution, stable candidate identities, and typed `t -> t+1` targets.
Edge losses use only persistent candidate identities; satellite flow remains a
node-domain target.

The trainer supports:

- one-step recurrent teacher forcing;
- scheduled sampling with explicit constant, linear, or cosine probability;
- multi-step on-policy rollout in `PaperAlignedLEOEnv`, where each requested
  action passes through the simulator gate and changes subsequent load,
  association history, candidates, and targets;
- truncation of recurrent gradients without replacing the executed transition;
- the typed gamma, log-intensity, flow, and feasibility loss;
- an additive DA-GWM decision-aware ranking loss;
- validation-only best-checkpoint selection and automatic `best`/`last`
  checkpoint writing.

The Snapshot trainer/evaluator use parallel typed contracts for gamma,
feasibility margin and instantaneous admitted load. They never route Snapshot
tensors through `PolicyDescriptors` or the Intensity--Flow target/loss. D0 and
new candidates require the configured Snapshot initializer, and predictions
are staged from `t` to `t+1` before they can affect an action.

## Method implementations

The method registry is:

| Name | Implementation |
|---|---|
| `tgn_mlp` | paper TGN with MLP messages |
| `tgn_kan` | paper TGN with KAN messages |
| `tgn_physick` | two-layer paper TGN with a learned vector-kernel bank and signed-L1 mixing |
| `ltt_r` | long-context graph front-end plus causal Transformer |
| `da_gwm` | PyG `GATv2Conv`, recurrent node state, and decision-aware loss |
| `s4` | 128-d Paper PhysiCK graph/readout with its GRU update replaced by official `state-spaces/s4` `S4Block` |
| `mamba2` | the same Paper PhysiCK graph/readout with official `mamba_ssm.Mamba2` temporal mixing |
| `conformer` | the same Paper PhysiCK graph/readout with official `torchaudio.models.Conformer` temporal mixing |
| `big_mlp` | shared two-layer, 128-d Paper TGN/GRU/readout; only each two-linear-layer MLP message operator is widened toward the reported 2.1M runtime-table class |
| `edge_attn` | shared two-layer, 128-d Paper TGN/GRU/readout; only the message operator is replaced by edge-conditioned multi-head graph attention, with a reported 2.1M runtime-table target |
| `temp_trans` | shared two-layer, 128-d MLP graph/readout; only GRU temporal mixing is replaced by a per-node message-sequence Transformer, with a reported 2.4M runtime-table target |
| `snapshot_mlp` | Snapshot-only graph/input/head with MLP messages; default factorial cell |
| `snapshot_physick` | Snapshot-only graph/input/head with the bounded signed PhysiCK operator; default factorial cell |
| `snapshot_ltt_r` | LTT-R backbone with the Snapshot head and H=10 action-coupled rollout loss; default matrix |
| `snapshot_da_gwm` | DA-GWM backbone with Snapshot head and Snapshot decision-aware action-coupled loss; default matrix |

LTT-R and DA-GWM are controlled experiment implementations described by this
paper, not names of installable upstream packages. The remaining external
backends are never replaced by a local toy fallback. Missing dependencies fail
at construction with an actionable error.

The three temporal-mixer rows inherit `model.physick`, the two directed
candidate-graph message layers, `mem_dim=msg_dim=emb_dim=128`, node injection,
and the global `intensity_flow` head from the root configuration. Their
`paper_baseline` entries may configure only the named temporal backend; the
factory rejects the previous independent width-256 edge-conditioned graph
path. The resolved architecture, `intensity_flow` interface and
`paper_physick` message-operator tags are stored with the checkpoint config.

The three runtime-table controls are similarly scope-checked at construction.
`big_mlp` and `edge_attn` keep the shared GRU memory and replace only the named
message operator; `temp_trans` keeps the ordinary MLP graph operators and
readout and replaces only temporal mixing. Their resolved checkpoint configs
record `architecture`, `shared_backbone`, `replacement_scope`, the reported
rounded capacity target and its provenance; the trainer checkpoint metadata
separately records the exact resolved trainable-parameter count. The
replacement-internal widths, head count, Transformer depth and context are
estimates because the manuscript does not report them. The targets are taken
from the SI runtime table itself, not from the smaller current reconstructed
PhysiCK implementation.

The registry, shell and dedicated Snapshot entry points now cover the primary
Snapshot+MLP and Snapshot+PhysiCK factorial cells. Snapshot+LTT-R and
Snapshot+DA-GWM are also constructible/trainable/evaluable and are included
in the default full method matrix
because their selected score weights are not reported and the executable config
labels those weights as estimates.

Snapshot contains no Intensity--Flow congestion memory. Its admitted-load
descriptor is the current executed admitted count divided by fixed satellite
capacity; its real-valued margin combines that instantaneous occupancy with the
current gamma cue. Neither field reads `L_t`, `Phi`, integrated intensity or the
EMA update. The simulator's boolean feasibility mask remains an authoritative
execution/hard-mask channel rather than a learned Snapshot descriptor.

Official dependency sources:

- S4: <https://github.com/state-spaces/s4>
- Mamba/Mamba2: <https://github.com/state-spaces/mamba>
- PyTorch Geometric: <https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html>
- torchaudio Conformer: <https://docs.pytorch.org/audio/main/generated/torchaudio.models.Conformer.html>

Mamba2's optimized GPU path requires a compatible Linux, PyTorch, CUDA and
compiler stack. Torchaudio must match the installed PyTorch binary. The S4
repository is pinned by source revision and placed on `PYTHONPATH` because it
does not expose the import used here as a stable PyPI distribution.

## Controllers, calibration, shield, and audits

`leo_pg.paper.controllers` contains stateful A3, CHO, and load-aware greedy
controllers plus an auditable sweep registry. Simulator feasibility and
capacity remain authoritative for all of them.

`leo_pg.paper.calibration` contains Cox multiplicative calibration,
covariate-error Cox envelopes, split-conformal upper calibration, veto shields,
and a soft risk-adjusted fixed-rank policy. `delta=1` is inert; smaller values
are explicitly labelled shield analyses.

`leo_pg.paper.audit` computes switch/no-switch shrink-jump samples, nearest-rank
quantiles, `I_sj = beta * alpha**D`, expansive fractions, and dwell sweeps. Its
output is an empirical diagnostic, not a stability theorem.

Calibration is fitted from an explicit tensor bundle and persisted separately:

```bash
python scripts/paper_calibrate.py \
  --input artifacts/calibration_inputs.pt \
  --kind covariate_error \
  --out artifacts/calibration/cox_envelope.json \
  --beta-l2-norm 1.2206556 \
  --quantile 0.95
```

The evaluator can load that file with
`leo_pg.paper.calibration:load_calibrator_factory`.

The one-command pipeline expands the complete A3/CHO/load-aware grid once on
the first selected method by default (`CONTROLLER_SWEEP=0` skips it).
`EVAL_RISK_SHIELD=1` enables the configured optional shield, and
`EVAL_SHRINK_JUMP_AUDIT=1` for the quantile/dwell audit.

## Parameter provenance

The runnable configuration is `configs/paper_protocol.yaml`. Every value is
classified as either manuscript-reported or estimated in
`PAPER_PARAMETER_AUDIT.md`.

Reported values include the 1,000-satellite/128-user setting, 100 ms control
period, Top-6 geometry candidates, capacity 40, score weights 1/0.4/0.6,
10-step dwell, 16 PhysiCK kernels, two message-passing layers, 50 epochs, the A3
grid, five runs, and 10,000 paired bootstrap resamples. The Snapshot SI search
uses gamma weights `{0.5, 0.75, 1.0, 1.25, 1.5}` and feasibility/load weights
`{0.2, 0.4, 0.6, 0.8}`; the selected primary cells are
Snapshot+MLP `(1.0, 0.6, 0.8)` and Snapshot+PhysiCK `(1.0, 0.4, 0.8)`.

The original integer seeds, TLE phases, exact `Phi(L)`, Cox baseline hazard and
normalization, D0/new-edge initializer, loss weights, CHO timing, and
load-to-dB conversion were not recoverable. The configuration supplies
conservative, explicit starting values for those fields. They are reasonable
for completing the software contract but are not represented as recovered
publication parameters.

The UAV/shared-service path is implemented under `leo_pg.uav_shared`, with
`scripts/uav_generate.py`, `scripts/uav_train.py`, and
`scripts/uav_evaluate.py` as its formal entry points. It includes the typed
graph/target contract, sharded five-run dataset, recurrent descriptor head,
one-step and action-coupled objectives, independent per-run checkpoints,
six-way paired substitution, held-out one-step loss, platform outcomes,
decision fidelity, and hierarchical run/episode bootstrap. The remaining
second-platform work is the full interface/operator factorial, stress-grid
orchestration, and manuscript display generation.
