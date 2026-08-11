# Manuscript implementation coverage

The repository contains executable, deliberately separate software paths for
the implemented NTN Intensity--Flow and Snapshot protocols. It also contains
an isolated action-coupled data/model/training/evaluation path for the
manuscript's UAV/shared-service platform. It is not yet a complete numerical reproduction and remains separate
from the legacy frozen-trajectory prototype.

## Implemented paper path

- simulator-authoritative and controller-facing descriptor channels;
- geometry-only Top-k candidates, deterministic identities, Protocol A/B hard
  mask selection, simulator feasibility/capacity admission, and keyed
  action-coupled transitions;
- p10 cue, EMA flow, frozen-gate Cox integration, fixed-rank controller, dwell,
  hysteresis, explicit D0/new-edge initialization, and staged `D_t -> a_t ->
  D_{t+1}` reuse;
- model/oracle/gamma/intensity/flow/intensity-flow substitutions confined to
  the current policy evaluation copy;
- typed tensor-only Intensity--Flow trajectory generation with immutable split
  manifests and persistent-edge next-step targets;
- one-step training, scheduled sampling, recurrent TBPTT, genuinely
  action-coupled multi-step rollout, DA-GWM decision loss, validation-selected
  best checkpoints, rolling last checkpoints, and resume metadata;
- TGN-MLP, TGN-KAN, the current Intensity--Flow PhysiCK operator, LTT-R and
  DA-GWM; plus official S4, Mamba2 and torchaudio Conformer temporal mixers
  that retain the same 128-dimensional Paper PhysiCK graph/message/readout
  contract instead of using an independent graph front-end;
- BigMLP and EdgeAttn runtime-table controls that retain the shared two-layer,
  128-d Paper TGN/GRU/Intensity--Flow contract and replace only the named
  message operator; plus TempTrans, which retains the standard MLP graph and
  readout and replaces only GRU temporal mixing. Their unreported internal
  dimensions are explicit estimates validated against the SI table's rounded
  2.1M/2.1M/2.4M capacity classes;
- A3, CHO, load-aware greedy, full controller grids, action-coupled paired
  evaluation, event traces, paper outcome/decision metrics, and paired
  bootstrap primitives;
- Cox and split-conformal calibration, optional veto/downweight shield,
  shrink-jump quantiles, dwell sweeps, and empirical `I_sj` diagnostics;
- a separate formal Snapshot-oracle dataset kind and shard generator, isolated
  Snapshot graph/target/head/loss/controller contracts, scheduled/action-coupled
  training adapter, staged model/oracle evaluator, checkpoint/provider path and
  shell dispatch;
- `snapshot_mlp` and `snapshot_physick` as the default interface-by-operator
  factorial cells, using the SI-selected score weights; plus default-matrix
  `snapshot_ltt_r` and `snapshot_da_gwm` cells whose unreported selected policy
  weights remain explicit estimates;
- strict rejection of diagnostic Intensity--Flow-to-Snapshot conversions as
  formal factorial data, so Snapshot training cannot silently inherit the
  Intensity--Flow behaviour policy/state distribution;
- a memoryless Snapshot load/margin path based on current executed admitted
  counts, fixed capacity and current gamma, with no read of integrated
  intensity, `L_t`, `Phi` or the EMA update;
- the UAV/shared-service protocol: validated state/action/result types,
  bounded-turn motion and energy, Top-3 stations, two service slots, FIFO
  queues, service completion/backfill, descriptor separation, deterministic
  randomness, snapshot/restore/fork, typed 9/6/6 graph features, stable-edge
  next-step targets, sharded multi-run generation, a recurrent descriptor head,
  one-step/scheduled/action-coupled training, per-run best/last checkpoints,
  six-way substitution evaluation, decision/outcome metrics, held-out one-step
  loss, and run/episode hierarchical paired bootstrap;
- a one-command dependency -> data -> method-matrix training -> checkpoint ->
  paired-evaluation script.

See `PAPER_PROTOCOL.md` for commands and `PAPER_PARAMETER_AUDIT.md` for the
reported-versus-estimated parameter registry.

## Still required for numerical manuscript reproduction

- the original TLE snapshot and phase state, GPWv4 mapping inputs, exact
  `Phi(L)`, link-budget conversion, Cox baseline/scales, D0/new-edge initializer,
  multi-task weights, integer seed list, and frozen publication configs;
- the lost training/evaluation trajectories, model checkpoints, optimizer
  histories, run manifests, event logs, bootstrap units, and unrounded source
  arrays;
- the original unreported runtime-baseline widths/depths/heads/context and
  unrounded parameter counts selected on the original validation units;
- manuscript-specific figure/table rendering and provenance manifests;
- the complete UAV four-cell interface/operator factorial, MLP/KAN/PhysiCK
  operator family, density/capacity sweep orchestration, and manuscript-specific
  second-platform figure/table renderer;
- any other named method not present in `leo_pg.paper.models.PAPER_METHODS`.

`configs/paper_protocol.yaml` supplies explicit engineering values where the
original settings are unavailable. Those values make the implemented path
executable; they do not fill the method gaps above or recover the lost
numerical experiment.

## Legacy path

The older `scripts/gen_data.py`, `scripts/train.py`, `scripts/eval.py`, and
`scripts/rollout.py` remain a compact prototype. They do not invoke the paper
environment or paper trainer. Use the `paper_*` entry points for manuscript
protocol work.
