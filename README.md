# Cox-PhysiCK

Physics-conditioned temporal graph learning for risk-aware handover in ultra-dense LEO non-terrestrial networks.

This repository contains the implementation, the final manuscript results, and the data needed to recompute their statistics. Start with [the result tables](results/tables/), [the figures](results/figures/), or [the manuscript](paper/paper.pdf).

## Repository layout

| Path | Contents |
|---|---|
| `src/cox_physick/` | Physical environment, controllers, neural models, training, evaluation, metrics, and plotting |
| `configs/paper.yaml` | Simulation and training configuration |
| `data/results_summary.csv` | One authoritative numeric summary of the final evaluation conditions |
| `data/raw/` | Executed associations, requests, and physical rates for the reported experiments |
| `data/episode_results.csv` | Episode-level counts and metrics supporting the summary |
| `data/calibration/` | Geometric entries, exposure, retention inputs, and diagnostic windows |
| `data/figure_points/` | Numeric coordinates and uncertainty values used in Figures 3–5 |
| `data/validation/` | Validation comparisons reported in the manuscript |
| `results/tables/` | Numeric CSV tables corresponding to the manuscript and its supporting analyses |
| `results/figures/` | PDF, SVG, and PNG result figures |
| `paper/` | Manuscript, bibliography, and its five figures |
| `docs/` | Data conventions, model specification, and physical execution rules |
| `tests/` | Mathematical, numerical, and integration checks |

## Install

Use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cox-physick inventory
```

The command is also available as `python -m cox_physick`.

## Reproduce the final results

```bash
cox-physick reproduce --full --output outputs/reproduced
```

This reduces the recorded execution arrays, checks the episode and condition statistics, reconstructs the calibration diagnostic, and exports the tables and figures. It does not require a trained checkpoint. The input files in `data/` remain unchanged.

Individual commands:

```bash
cox-physick audit --full --output outputs/audit
cox-physick calibrate --output outputs/calibration
cox-physick tables --output results/tables
cox-physick plots --output results/figures
```

[Data and metric definitions](docs/results.md) explain the statistical units, denominator conventions, and stored association encoding. Learned-controller standard deviations are calculated across training-seed means; deterministic-controller standard deviations are calculated across test episodes.

## Run physical simulations

```bash
cox-physick evaluate --method greedy_analytic --satellites 2000 --users 100 \
  --horizon 800 --prefixes 200 800 --episodes 30 --seed 20001 \
  --output runs/greedy
```

Rule controllers are `maxsinr_ttt`, `maxrst`, `greedy_analytic`, and `cox_only`. The simulator records geometry, beam schedules, weather, requests, admission, execution, rates, and cost components. See [the environment specification](docs/environment.md).

## Train and evaluate learned controllers

```bash
cox-physick train --method full --seed 1 --device cpu --output runs/full_seed1
cox-physick evaluate --method full --checkpoint runs/full_seed1/selected.pt \
  --seed 20001 --episodes 30 --prefixes 200 800 --output runs/full_seed1_test
```

Use `--device cuda` on a CUDA host. Available learned methods are `full`, `mlp213`, `kan_generic`, `leo_madrl`, `no_triplet`, `no_cox_rst`, `eph_physick`, and `mlp_coeff`. TGN-MLP and the no-kernel ablation share the same implementation and result records. The adapted LEO-MADRL comparator uses parameter-sharing hysteretic DQN. Architecture and training details are in [models.md](docs/models.md).

The default residual-model budget is four rounds of 80 episodes and 5,000 optimizer updates per round. DQN uses 256,000 system epochs. A complete study can be generated and executed with:

```bash
cox-physick plan --output outputs/experiment_plan.json
cox-physick run-plan --plan outputs/experiment_plan.json --output runs/study --device cuda
cox-physick summarize-runs --runs runs/study --output outputs/new_results
```

Use `--stage train` or `--stage evaluate`, together with `--job JOB_ID`, to select a particular job. New runs have their own output records; the supplied paper results remain a fixed dataset. The executable environment specifies resource and weather rules that are not recoverable from the stored association/rate arrays alone; reproducing their statistics is distinct from re-running historical training and physical traces.

## Tests

```bash
python -m pytest tests -q
```

The delivered numerical checks are summarized in `results/verification.json`.

## Manuscript and citation

The paper is **Cox-PhysiCK: Physics-Conditioned Temporal Graph Learning for Risk-Aware Handover in Ultra-Dense LEO-NTNs**. Citation metadata is in `CITATION.cff`.

To compile the manuscript, run `latexmk -pdf main.tex` inside `paper/`. The code retains the MIT license from the original repository.
