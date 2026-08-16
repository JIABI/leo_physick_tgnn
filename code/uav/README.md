# UAV shared-service platform

This package implements the action-coupled UAV service-allocation platform:
bounded-turn motion, finite energy, station queues and slots, Top-3 candidate
construction, controller-facing descriptor staging, MLP/KAN/PhysiCK world
models, proposal/execution dynamics and paired closed-loop inference.

The experiment matrix covers operator comparison, density multipliers 1--5,
capacity factors 1.0/0.75/0.5, four structural perturbations, descriptor
substitution and the full component-removal ladder. Every evaluation resolves
five distinct training checkpoints and 30 held-out episodes per run.

Install the package:

```bash
python -m pip install -e .
```

Inspect the complete experiment command matrix:

```bash
PYTHONPATH=. python cli/run_experiments.py \
  --release-config configs/experiment_release.template.yaml \
  --output-root /path/to/uav_plan \
  --dry-run
```

Individual entry points:

```bash
PYTHONPATH=. python cli/generate.py --help
PYTHONPATH=. python cli/train.py --help
PYTHONPATH=. python cli/evaluate.py --help
PYTHONPATH=. python cli/build_manifest.py --help
```

`uav_cfs/experiments.py` is the authoritative task registry.
`PAPER_TO_CODE_MAP.md` links each paper object to its implementation, and
`RESULT_SCHEMA.md` describes run-by-episode identity and metric fields.

