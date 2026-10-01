# Cox-PhysiCK: TCOM revision and executable code

This local release contains the TCOM revision, its supplied results, and an
executable implementation of the Cox-PhysiCK handover controller. The upstream
satellite/UAV project is preserved; use **`code/tcom`** for this paper. The
previous root documentation is retained in [README.upstream.md](README.upstream.md).

## Contents

| Path | Content |
|---|---|
| `manuscript/tcom/` | Revised LaTeX, PDF, bibliography, figures and change notes |
| `code/tcom/` | Physical environment, controllers, recurrent models, training, evaluation, metrics and archive tools |
| `data/tcom/` | All current supplied results, 315 raw execution files, 115-condition index and calibration records |
| `results/tcom/` | Recomputed audits, editable plots and a complete experiment plan |
| `docs/TCOM_ENVIRONMENT.md` | Candidate subband, beam, weather, admission and execution rules |
| `docs/TCOM_MODELS.md` | PhysiCK/TGN/KAN/DQN implementations and training interfaces |
| `docs/TCOM_RESULTS.md` | File map, units, statistical aggregation and replay commands |
| `docs/TCOM_REVISION.md` | Necessary manuscript edits and remaining historical mapping |

## Install

Python 3.10 or later, NumPy, SciPy, PyTorch and SGP4 are required. No orbital
network download is needed for the synthetic Walker constellation.

```bash
python -m venv .venv-tcom
source .venv-tcom/bin/activate
python -m pip install -e 'code/tcom[test]'
cox-physick inventory
```

The package can also be invoked with `python -m cox_physick`.

## Recompute the supplied results

```bash
cox-physick reproduce --full --output results/tcom/reproduced
```

This reads the supplied arrays, recomputes episode metrics and their declared
aggregation, checks the cost/load identities, replays the recorded calibration
entries and thinning marks, and regenerates the three numerical figures.
It does not train models or replace the supplied outcomes. The input archive
is unchanged. Fresh executions are written to a separate output directory.

For individual tasks:

```bash
cox-physick audit --full --output results/tcom/audit
cox-physick calibrate --output results/tcom/calibration
cox-physick plots --output results/tcom/figures
```

## Execute physical controllers

```bash
cox-physick evaluate --method greedy_analytic --satellites 2000 --users 100 \
  --horizon 800 --prefixes 200 800 --episodes 30 --seed 20001 \
  --output runs/tcom/greedy_N2000_K100
cox-physick evaluate --method cox_only --satellites 2000 --users 500 \
  --horizon 800 --prefixes 200 --episodes 30 --seed 20001 \
  --output runs/tcom/cox_only_N2000_K500
```

Each run saves the effective configuration, executed associations, requests,
rates, occupancy, physical diagnostics, exact geometry, beam trajectories and
weather. Environment seed 10001 is used for validation and 20001 for testing;
all methods and training seeds share the same exogenous test episode IDs.

## Train and evaluate learned controllers

```bash
cox-physick train --method full --seed 1 --device cpu \
  --output runs/tcom/full_seed1
cox-physick evaluate --method full --checkpoint runs/tcom/full_seed1/selected.pt \
  --seed 20001 --episodes 30 --prefixes 200 800 \
  --output runs/tcom/full_seed1_test
cox-physick train --method leo_madrl --seed 1 --device cpu \
  --output runs/tcom/madrl_seed1
```

Use `--device cuda` on a CUDA host. Residual training implements four rounds,
80 episodes and 5,000 updates per round; DQN uses 256,000 system epochs. These
are substantial experiments. No pretrained model is silently substituted for
training. `--resume` resumes an explicitly supplied training-state file.
All numerical claims in the manuscript continue to use the supplied archive.

Available controls: `maxsinr_ttt`, `maxrst`, `greedy_analytic`, `cox_only`,
`mlp213`, `kan_generic`, `leo_madrl`, `full`, `no_triplet`, `no_cox_rst`,
`eph_physick`, and `mlp_coeff`. The no-kernel ablation aliases `mlp213`.

## Run all recorded conditions

```bash
cox-physick plan --output results/tcom/experiment_plan.json
cox-physick run-plan --plan results/tcom/experiment_plan.json \
  --output runs/tcom/study --device cuda
```

The plan contains 190 training jobs and 315 evaluation jobs. Prefixes share a
single 800-step trajectory, density shifts reuse the corresponding K=100 model,
hysteresis sweeps reuse weights, and the 17 weight settings have separate
training jobs. `run-plan --stage train|evaluate --job JOB_ID` selects work.

## Validation

```bash
python -m pytest code/tcom/tests -q
```

See `results/tcom/verification.json` for the checks actually run for this
release. Short training/rollout tests validate executable paths and are kept
separate from the manuscript experiment results.

The release specifies previously underspecified band, beam and weather rules
in its environment configuration. Their implementation is complete; assigning
those choices to historical result records requires the corresponding author
confirmation. The source-result calculations are reproducible independently
of that assignment. See [the revision notes](docs/TCOM_REVISION.md).

Aggregate newly executed runs with the same statistical units:

```bash
cox-physick summarize-runs --runs runs/tcom/study \
  --output results/tcom/new_summary
```

This produces wide and long condition summaries and cost contrasts against
greedy. Duplicate episode prefixes and mismatched test panels are rejected.
For Anaconda installations with a failing system `readline` extension, tests
can run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:capture code/tcom/tests -q`.
