# Controller-facing state for feedback-coupled allocation

This repository is the software companion to *Controller-facing state design
for feedback-coupled world-model control*. It implements the satellite and UAV
platforms with one shared experimental contract: five independently
initialized training runs, 30 paired held-out episodes per run, equal weighting
of run means, and run-first paired hierarchical bootstrap intervals.

The repository contains source code, experiment registries, configuration
templates, manifest schemas and aggregation tools. Experiment outputs are
written to user-selected directories outside the source tree.

## Layout

- `code/satellite`: LEO simulator, candidate graphs, Snapshot and
  Intensity--Flow interfaces, MLP/KAN/PhysiCK models, controls and the complete
  satellite study planner.
- `code/uav`: UAV service simulator, typed graph world model, the three message
  operators, descriptor substitutions, stress studies and experiment
  dispatcher.
- `code/shared`: run, episode and result records plus paired inference tools.
- `configs`: cross-platform study registry, analysis plan and manifest schemas.
- `scripts`: manifest generation, result collection, aggregation and identity
  checks.

## Installation

Python 3.10 or newer is supported. Install each platform in editable mode from
the repository root:

```bash
python -m pip install -e "code/satellite[paper]"
python -m pip install -e code/uav
```

See `ENVIRONMENT.md` for dependency details and the optional S4/Mamba2 setup.

## Satellite studies

Inspect the active methods and all 13 registered studies:

```bash
cfs-satellite inventory
```

Copy `code/satellite/configs/author/paper_artifacts.template.yaml`, enter the
five seed bundles, paired episode identities, ephemeris inputs and study
parameters, then create the full command plan:

```bash
cfs-satellite plan \
  --config code/satellite/configs/paper_protocol.yaml \
  --studies code/satellite/configs/studies.yaml \
  --author /path/to/paper_artifacts.yaml \
  --select all \
  --out /path/to/satellite_plan
```

Planning writes resolved configurations and commands but does not launch a
study. After inspecting the plan, execution is explicit:

```bash
cfs-satellite run-plan \
  --plan /path/to/satellite_plan/satellite_plan.json \
  --execute
```

## UAV studies

The UAV dispatcher covers 46 tasks across operator comparison, density,
capacity, structural perturbation, descriptor substitution and component
removal. Review `code/uav/configs/experiment_release.template.yaml`, set the
dataset and checkpoint locations, and inspect the complete command matrix:

```bash
PYTHONPATH=code/uav python code/uav/cli/run_experiments.py \
  --release-config code/uav/configs/experiment_release.template.yaml \
  --output-root /path/to/uav_plan \
  --dry-run
```

The individual data, training and evaluation entry points are:

```bash
PYTHONPATH=code/uav python code/uav/cli/generate.py --help
PYTHONPATH=code/uav python code/uav/cli/train.py --help
PYTHONPATH=code/uav python code/uav/cli/evaluate.py --help
```

Each UAV evaluation resolves one checkpoint per training seed and uses exactly
30 held-out episodes for each of the five runs.

## Run-level records and inference

The shared tools preserve `q_t` proposal events separately from executed
actions `a_t`, keep undefined ratios as undefined, and bind every result row to
its platform, condition, cell, method, run, checkpoint, frozen configuration
and code identity.

```bash
python scripts/make_episode_manifest.py --help
python scripts/collect_run_episode_results.py --help
python scripts/summarize_results.py --help
python scripts/audit_release.py --help
```

`scripts/summarize_results.py` computes equal-run-weight summaries and paired
run-first bootstrap contrasts from canonical run-by-episode records. The exact
paper-to-module correspondence is listed in `PAPER_TO_CODE_MATRIX.md`.
