# Paper-to-code matrix

## Shared experimental contract

| Paper object | Implementation |
|---|---|
| Five independent model/optimizer/data-order seed bundles | `code/shared/schemas.py::RunManifest`; both training entry points |
| Thirty matched held-out episodes per run | `code/shared/schemas.py::EpisodeManifest`; platform dataset manifests |
| Paired episode and exogenous streams | `code/shared/manifest.py`; platform evaluation modules |
| Proposal `q_t` versus execution `a_t` | satellite `sim/paper_environment.py`; UAV `policy.py` and `environment.py` |
| Undefined denominator handling | `code/shared/schemas.py::MetricObservation`; platform metric modules |
| Equal weighting of run means | `code/shared/statistics.py::summarize_condition` |
| Paired run-first hierarchical bootstrap | `code/shared/statistics.py::paired_hierarchical_bootstrap` |
| Canonical run-by-episode export | `scripts/collect_run_episode_results.py` |
| Exact result/checkpoint/configuration/code identity join | `scripts/audit_release.py` |

## Satellite platform

| Paper method or study | Implementation |
|---|---|
| Dynamic user-satellite graph and Top-6 candidates | `code/satellite/src/leo_pg/graph/`; `sim/paper_environment.py` |
| Snapshot interface | `paper/snapshot.py`, `snapshot_data.py`, `snapshot_models.py` |
| Cox Intensity and EMA Flow | `physics/cox_intensity.py`, `physics/descriptors.py`, `sim/intensity_flow.py` |
| MLP, KAN and PhysiCK message operators | `kernels/mlp.py`, `kernels/kan.py`, `kernels/physick/` |
| Four-cell crossed closed-loop study | `configs/studies.yaml::FCT-CL`; `paper/release_cli.py` |
| Decision diagnostics and stress sweep | `DEC-K`, `STRS-K`; decision and closed-loop runners |
| 300-second K=500 trajectory | `LONG-HORIZON-300S`, 3,000 control steps |
| Dwell sensitivity | `FCT-DWELL`, values 5, 10, 20 and 40 |
| Public ephemeris replay | `EPH-REPLAY`; `sim/ephemeris.py` |
| Descriptor substitution | `ORACLE-LADDER`; `eval/substitution.py` |
| Component and temporal ablations | `ABLATIONS`; S4/Mamba2 are exploratory entries |
| Teacher forcing, scheduled sampling and H=5/10/20 | `ROLLOUT-AWARE`; `train/rollout.py` |
| Score-weight sensitivity | `SCORE-WEIGHT-SENSITIVITY` |
| Snapshot validation-grid selection | `SNAPSHOT-WEIGHT-SELECTION`; `paper_snapshot_select_weights.py` |
| LTT-R and DA-GWM controls | `paper/baselines.py`; `LEARNED-CONTROLS` |
| A3, CHO and load-aware controls | `paper/controllers.py`; `CLASSICAL-CONTROLS` |

## UAV platform

| Paper method or study | Implementation |
|---|---|
| Motion, energy, station queues, slots and service | `code/uav/uav_cfs/state.py`, `environment.py` |
| Reachability and Top-3 candidate graph | `environment.py::_candidate_edges`; `graph.py` |
| Local utility, Flow and composite Intensity | descriptor functions in `environment.py` |
| MLP/KAN/PhysiCK operators | `operators.py`, `model.py` |
| Proposal, hysteresis and simulator execution | `policy.py`, `environment.py::step_action` |
| Operator comparison | `experiments.py::paper_task_matrix` |
| Density multipliers 1--5 | `paper_task_matrix`; `cli/evaluate.py` |
| Capacity factors 1.0, 0.75 and 0.5 | `paper_task_matrix`; `cli/evaluate.py` |
| Four structural perturbations | `perturbations.py`; `structural_mismatches.template.yaml` |
| Partial and complete descriptor substitution | `evaluation.py::UAVSubstitutionMode` |
| Component removals | `experiments.py::UAVAblation`; model configuration files |
| Five-run, 30-episode dispatch | `cli/run_experiments.py`; `cli/evaluate.py` |
