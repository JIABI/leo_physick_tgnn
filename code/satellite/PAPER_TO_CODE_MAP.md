# Manuscript v8 to satellite code map

| Manuscript object / evidence | Primary implementation | Runner / output |
|---|---|---|
| Candidate incidence graph, elevation mask, Top-6 and identity tie break | `src/leo_pg/graph/candidates.py`, `src/leo_pg/sim/paper_environment.py` | `scripts/paper_generate.py` |
| Public TLE propagation and replay geometry | `src/leo_pg/sim/ephemeris.py`, `PaperAlignedLEOEnv._build_ephemeris` | `EPH-REPLAY` |
| Simulator state and typed observation/action/result records | `src/leo_pg/sim/state.py`, `src/leo_pg/sim/protocol.py` | every action-coupled study |
| Proposed target `q_t` and executed association `a_t` | `src/leo_pg/control/policy.py`, `PaperAlignedLEOEnv.step_action` | serialized episode shards from `paper_evaluate.py` |
| Cox first-violation Intensity and finite closed-gate zero | `src/leo_pg/sim/intensity_flow.py` | Intensity--Flow generator/evaluator |
| EMA Flow and feedback-coupled load transition | `src/leo_pg/sim/intensity_flow.py`, `PaperAlignedLEOEnv.step_action` | Intensity--Flow generator/evaluator |
| Controller-facing descriptor copy and partial substitution | `src/leo_pg/eval/substitution.py`, `src/leo_pg/paper/evaluation.py` | `FCT-CL`, `ORACLE-LADDER` |
| Fixed rank score, dwell, hysteresis and execution authority | `src/leo_pg/control/policy.py`, `src/leo_pg/sim/paper_environment.py` | all closed-loop studies |
| Snapshot interface, margin/load fields and isolated dataset | `src/leo_pg/paper/snapshot.py`, `snapshot_data.py` | `paper_snapshot_generate.py` |
| Snapshot validation score-weight grid | `snapshot.py:SNAPSHOT_SI_SCORE_GRID` | `paper_snapshot_select_weights.py`, `SNAPSHOT-WEIGHT-SELECTION` |
| MLP, KAN and PhysiCK message operators | `src/leo_pg/kernels/`, `src/leo_pg/models/tgn/tgn.py` | `FCT-CL`, `DEC-K`, `STRS-K` |
| PhysiCK kernel bank and signed-l1 projection | `src/leo_pg/kernels/physick/` | TGN--PhysiCK model factory |
| LTT-R and DA-GWM | `src/leo_pg/paper/baselines.py` | `LEARNED-CONTROLS` |
| S4 and Mamba2 temporal swaps | `src/leo_pg/paper/backbones.py`, `baselines.py` | exploratory rows of `ABLATIONS` |
| One-step, scheduled-sampling and action-coupled training | `src/leo_pg/paper/training.py`, `snapshot_training.py` | `paper_train.py`, `paper_snapshot_train.py`, `ROLLOUT-AWARE` |
| Fixed-checkpoint score-weight perturbations | rank policy + action-coupled evaluator | `SCORE-WEIGHT-SENSITIVITY` |
| Next-step Top-1, Kendall, near-tie flips and native regret | `scripts/paper_decision_evaluate.py`, `src/leo_pg/paper/metrics.py` | `DEC-K` JSON per run/method/K |
| Zero-shot reported stress sweep | author-provided `stress_index_overrides` | `STRS-K` |
| 300-s K=500 trajectories | standard action-coupled evaluator with 3,000 steps | `LONG-HORIZON-300S` |
| Dwell sensitivity | policy override only | `FCT-DWELL` at 5/10/20/40 |
| Component removals | environment/model hooks: `intensity.representation`, `flow.freeze`, `candidates.thinning_enabled`, `model.edge_feature_mask`, `model.temporal_memory`, `model.physick.use_kernel_bank` | exact author `ablation_overrides`, `ABLATIONS` |
| A3, CHO and load-aware greedy | `src/leo_pg/paper/controllers.py` | `CLASSICAL-CONTROLS` |
| Outcome, event, proposal/execution and hierarchical-summary primitives | `src/leo_pg/paper/metrics.py`, `src/leo_pg/train/system_metrics.py` | evaluation bundles and shared summarizer |
| Cox/conformal calibration utilities and inactive shield | `src/leo_pg/paper/calibration.py`, `scripts/paper_calibrate.py` | fitted JSON from author-supplied held-out calibration tensor bundle |
| Shrink--jump/dwell diagnostics | `src/leo_pg/paper/audit.py`, Snapshot metric adapter | evaluation bundle audit records |
| Five-run / 30-episode pairing and full study DAG | `src/leo_pg/paper/release_cli.py` | `satellite_plan.json` |

The publication registry is `src/leo_pg/paper/models.py`. It contains the
reported methods plus exploratory S4/Mamba2.
