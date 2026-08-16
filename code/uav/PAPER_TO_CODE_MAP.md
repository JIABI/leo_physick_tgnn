# UAV paper-to-code map

| v8 method/result object | Implementation |
|---|---|
| UAV state, motion, energy, queue, occupancy, slots | `uav_cfs/state.py`, `uav_cfs/environment.py` |
| Top-3 reachable candidate graph and stable identities | `uav_cfs/environment.py::_candidate_edges`, `uav_cfs/graph.py` |
| Local service utility | `uav_cfs/environment.py::_eta_all_stations` |
| Station Flow EMA | `uav_cfs/environment.py::_raw_station_pressure`, `_update_station_flow` |
| Composite Intensity containing Flow | `uav_cfs/environment.py::_descriptors` |
| Controller-facing copy versus simulator authority | `uav_cfs/state.py`, `uav_cfs/evaluation.py::run_uav_closed_loop` |
| Fixed rank score and hysteresis, no dwell | `uav_cfs/policy.py` |
| Proposal `q_t` and executed action `a_t` | `uav_cfs/policy.py`, `uav_cfs/environment.py::step_action` |
| TGN/GRU temporal model and descriptor heads | `uav_cfs/model.py` |
| MLP, KAN and PhysiCK edge-message operators | `uav_cfs/operators.py` |
| Signed-L1 gain control and its removal | `uav_cfs/operators.py::PhysiCKMessageOperator`, `configs/model_physick_no_gain_control.yaml` |
| One-step and rollout-aware objectives | `uav_cfs/training.py`, `cli/train.py` |
| Recursive staging and persistent-edge join | `uav_cfs/evaluation.py`, `uav_cfs/training.py` |
| Oracle and partial descriptor substitution | `uav_cfs/evaluation.py::UAVSubstitutionMode`, `apply_policy_substitution` |
| Nested no-Intensity/no-Flow/local-cue interventions | `uav_cfs/experiments.py::UAVAblationDescriptorProvider` |
| Density and capacity sweeps | `uav_cfs/experiments.py::paper_task_matrix`, `cli/evaluate.py` |
| Four structural mismatches | `uav_cfs/perturbations.py`, `uav_cfs/environment.py` |
| Physical and decision diagnostics | `uav_cfs/metrics.py` |
| Five-run x 30-episode matched evaluation | `uav_cfs/data.py`, `cli/evaluate.py`, `cli/build_manifest.py` |
| Run-first hierarchical bootstrap | `uav_cfs/metrics.py::hierarchical_paired_bootstrap` |
| Complete experiment dispatch | `cli/run_experiments.py` |

The four perturbation values are supplied through
`configs/structural_mismatches.template.yaml`; the dispatcher carries them into
each resolved task configuration.
