# Predictive accuracy alone does not determine feedback-coupled world-model control

This repository is the software companion to the manuscript *Predictive
accuracy alone does not determine feedback-coupled world-model control*. It
provides a compact controller-facing interface demonstration, a
machine-readable experiment registry and validation tools for the version 4
Zenodo source-data release.

Software version 2.0.0 is a new companion release built from the public
implementation lineage at upstream commit
`d096a2020649d360dfffc860309a510734ea6033`. That upstream commit predates this
release and does not contain the version 2.0.0 additions. The source-data
version series is available under Zenodo concept DOI
[`10.5281/zenodo.21966529`](https://doi.org/10.5281/zenodo.21966529).

## What this release reproduces

The repository separates three tasks that require different evidence and
compute.

| Level | Entry point | What it establishes |
|---|---|---|
| Semantic demonstration | `./run_demo.sh` | Runs a deterministic CPU example of candidate identity, controller-visible descriptors, score construction, hard feasibility filtering and execution. It illustrates the prediction-to-action contract; it does not reproduce manuscript effect sizes. |
| Zenodo source-data validation | `verify-source-data` | Checks package hashes, closed manifest/checksum path sets, CSV counts and shapes, and declared QC status. It does not reinterpret the release's wide source tables. |
| Statistical estimator implementation | `scripts/summarize_results.py` | Applies the registered platform-specific estimator to canonical long-form run-by-episode records produced by the research workflow. These records are a separate input contract, not the Zenodo v4 wide CSV layout. |
| Full training and experimental rerun | Not invoked by the root CLI | Requires original training inputs, checkpoints and experiment-specific assets that are not included in the root release workflow. |

The root commands are CPU-only and do not train a model.

## Quick start

From a fresh checkout:

```bash
./setup.sh
source .venv/bin/activate
./run_demo.sh
./run_tests.sh
./verify_paper_contract.sh
```

This is a source-repository workflow: the root package relies on the bundled
`code/satellite` and `code/uav` trees and is not distributed as a standalone
wheel. Use `setup.sh` from the repository checkout.

`setup.sh` creates an isolated `.venv` and installs the root package, the two
platform runtimes used by the demonstration and the test runner. It does not install
the optional satellite paper-training extras. Activating the environment, as
shown above, also makes the direct `python -m ...` examples below portable.
The demonstration writes only to
the repository's ignored output directory. See
[ENVIRONMENT.md](ENVIRONMENT.md) for manual installation and platform notes.
Without arguments, `run_tests.sh` runs the root release-contract suite followed
by the satellite and UAV suites.

## Command-line interface

The stable entry point is:

```bash
python -m controller_facing_state.cli --help
```

Its public subcommands are:

```bash
python -m controller_facing_state.cli demo --help
python -m controller_facing_state.cli experiments --help
python -m controller_facing_state.cli verify-config --help
python -m controller_facing_state.cli audit-model-identities --help
python -m controller_facing_state.cli verify-source-data --help
```

- `demo` runs the deterministic controller-facing-interface example.
- `experiments` lists the registered study identifiers; `--id FCT-CL` or
  `--id EXP1` (and the corresponding EXP2--EXP5 identifiers) prints a complete
  design record.
- `verify-config` checks the frozen values in the bundled manuscript protocol.
- `audit-model-identities` constructs the five Table S11 structural references,
  reports their real parameter counts and refuses exact frozen-ID claims when
  they differ. Add `--require-exact` when a recovered constructor is expected.
- `verify-source-data` checks an unpacked Zenodo v4 source-data directory and
  validates its package-level integrity and declared QC status.

Pass an unpacked Zenodo source-data directory to the contract script to run
both configuration and source-data checks:

```bash
./verify_paper_contract.sh ./source_data/ControllerFacingState_SourceData_v4.0.0
```

No release command downloads Zenodo data or experiment assets implicitly.

## Statistical estimator interface

The estimator implementation is platform-specific. Satellite summaries use ten
equal-weight training-run means with two-sided Student-t intervals (df=9), and
satellite contrasts use paired Student-t intervals over the ten matched run
means. UAV summaries and paired contrasts use five independent runs with the
nested-within-run hierarchical bootstrap.

This command consumes the canonical long-form schema written by
`scripts/collect_run_episode_results.py` from original evaluation bundles. It
does **not** accept the wide CSV tables in the Zenodo v4 archive directly. The
archive verifier and the statistical estimator are therefore independent
entry points. A populated analysis plan and canonical records are required;
the bundled JSON is a schema/template rather than a ready analysis.

Create one analysis plan per platform from
`configs/ANALYSIS_PLAN_TEMPLATE.json`, then call the estimator explicitly:

```bash
python scripts/summarize_results.py \
  --platform satellite \
  --estimator run_first_student_t \
  --input ./records/satellite_run_episode_results.csv \
  --analysis-plan ./analysis/satellite_plan.json \
  --output-dir ./analysis/satellite

python scripts/summarize_results.py \
  --platform uav \
  --estimator run_first_hierarchical_bootstrap \
  --input ./records/uav_run_episode_results.csv \
  --analysis-plan ./analysis/uav_plan.json \
  --output-dir ./analysis/uav
```

The command requires the CLI, plan and input records to name the same platform
and rejects an estimator assigned to the other platform. Output tables record
the estimator, run count and either degrees of freedom or bootstrap settings.

## Experiment registry

The registry records the design rather than only the reported point estimates.
It includes:

- FCT-CL: the core two-by-two interface-by-operator factorial, with ten
  independent runs and 30 held-out episodes per cell;
- EXP1: two interface contracts, 20 paired seeds per contract, 50 checkpoints
  per model, five validation-only selection rules, three near-tie strata and
  six nested reassociation windows;
- EXP2: the two-phase, eight-cell descriptor-by-staging-by-score-map crossover;
- EXP3: the four-arm operator-by-objective comparison with 20 paired seeds per
  arm;
- EXP4: the completed external-environment quantitative records, explicitly
  marked as provenance-incomplete in the source-data release; and
- EXP5: the ordered four-variant message-operator ablation, including the
  configuration-specific projection radius.

Use the CLI rather than copying values from this page:

```bash
python -m controller_facing_state.cli experiments
python -m controller_facing_state.cli experiments --id EXP1
```

The machine-readable registry is the design record exposed by this software
release.

## Implemented experiment surface

FCT-CL and EXP1, EXP2, EXP3 and EXP5 have executable protocol, analysis or
model-construction components, frozen configuration records and CPU tests.
For EXP2, the release implements typed current/causal staging and the two
score equations explicitly defined in the manuscript (D0 x S0 and D1 x S1).
The v4 tables do not preserve the field maps for D0 x S1 or D1 x S0. Those
cross-domain cells therefore fail closed unless a dated author mapping is
provided through the explicit adapter; this release does not claim to rerun
the historical eight-cell trajectories from the aggregate tables alone.
EXP4 remains a quantitative source-data record: its environment identity,
source version, trace split and field mapping were not recorded, so the
external evaluation cannot be rerun from this release.

The MRG-PPO comparator is implemented as a pure-PyTorch masked recurrent graph
actor and centralized training-only critic, together with its reward, PPO/GAE
primitives and frozen budget. The demo verifies 731,905 actor parameters and
687,233 critic parameters on a CPU forward pass. Its tuning grid and selected
configuration identifier are recorded, but the v4 ledger does not contain the
winning numeric learning-rate and entropy-coefficient pair. Those two fields
remain `null` in `configs/models/mrg_ppo.yaml`; exact MRG-PPO retraining requires
the corresponding locked run configuration.

The public LTT-R, DA-GWM and TGN-PhysiCK constructors are executable structural
references, not recovered copies of the original frozen checkpoints. Their
real parameter counts do not equal the five exact identities reported in
Supplementary Table S11. `leo_pg.paper.model_identity` records the ledger
targets and refuses to attach a frozen implementation ID unless a real
constructor matches exactly; it never pads a model with unused parameters.

## Repository layout

- `src/controller_facing_state/`: root demonstration, registry and verification
  package.
- `configs/manuscript_v4.yaml`: the shared controller, platform, model and
  statistical protocol.
- `configs/experiments_v4.yaml`: the FCT-CL and EXP1--EXP5 design registry.
- `configs/models/mrg_ppo.yaml`: the MRG-PPO architecture, reward, optimization
  and interaction-budget contract.
- `configs/`: manifest schemas and retained research-workflow templates.
- `code/satellite/`: satellite simulator and MLP, KAN and PhysiCK operator
  implementations inherited from the upstream research code.
- `code/satellite/src/leo_pg/paper/extended_experiments.py`: executable EXP1
  selection/equivalence rules, EXP3 objective and contrast rules, and EXP5
  operator/diagnostic implementations.
- `code/satellite/src/leo_pg/paper/crossover.py`: the typed EXP2 registry,
  staging runner, documented native score bindings and fail-closed adapter for
  author-supplied cross-domain field maps.
- `code/satellite/src/leo_pg/paper/model_identity.py`: strict Table S11
  parameter-ledger audit and exact-identity gate for structural references.
- `code/satellite/src/leo_pg/paper/mrg_ppo.py`: the MRG-PPO actor, critic,
  action distribution, reward and optimization primitives.
- `code/uav/`: UAV shared-service simulator and operator implementations.
- `code/shared/`: shared records, metrics, seeding and statistical utilities.
- `scripts/`: run-record collection and aggregation tools retained for the
  extended research workflow.
- `tests/`: root release-contract and deterministic demonstration tests;
  platform-specific suites are under `code/satellite/tests/` and
  `code/uav/tests/`.

The mapping from manuscript objects to executable modules is in
[PAPER_TO_CODE_MATRIX.md](PAPER_TO_CODE_MATRIX.md).

## Evidence boundary

The Zenodo archive publishes authoritative checkpoint-level and run-level
records for the supported analyses. Those records provide the evidence needed
for independent recalculation, but this repository does not silently convert
their experiment-specific wide schemas into canonical long records. The root
verifier establishes archive integrity. Checkpoint
tensors, state-level training streams and the complete external environment
used by EXP4 are not included.

## Citation and licence

Please cite the associated manuscript and the version-specific Zenodo record
used in an analysis. The concept DOI above resolves to the version series;
Zenodo assigns a separate DOI to each published version. Machine-readable
software citation metadata are provided in [CITATION.cff](CITATION.cff).

After the version 2.0.0 package is uploaded, freeze the release under a new
`v2.0.0` tag and cite the immutable commit resolved by that tag. The upstream
commit `d096a2020649d360dfffc860309a510734ea6033` records the pre-v2 lineage; it
must not be cited as the version 2.0.0 implementation.

The software in this repository is released under the [MIT License](LICENSE).
The Zenodo source-data archive retains its own CC BY 4.0 licence.
