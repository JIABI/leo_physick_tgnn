# Changelog

All notable changes to the public software companion are recorded here.

## 2.0.0 - pending release

- Updated the release title and metadata to match *Predictive accuracy alone
  does not determine feedback-coupled world-model control*.
- Added a root CPU-only command-line interface for a deterministic semantic
  demonstration, experiment-registry inspection, configuration checks and
  Zenodo source-data verification.
- Added machine-readable FCT-CL and EXP1--EXP5 design records and
  release-contract tests.
- Added the MRG-PPO actor, training-only critic, reward and PPO/GAE primitives,
  with an explicit record of the unresolved selected tuning pair.
- Added executable EXP1/EXP3/EXP5 components and a typed EXP2 staging and
  selection runner that fails closed when an unpublished cross-domain field
  map is required.
- Added a strict Table S11 parameter-identity ledger that reports the real
  structural-reference counts and refuses to relabel them as the original
  frozen checkpoint identities.
- Separated semantic demonstration, source-data validation and statistical
  recomputation inputs, and full training so that each command states the
  evidence it provides.
- Enforced satellite inference over ten run means with Student-t intervals
  (df=9) and UAV inference over five runs with the hierarchical bootstrap.
- Required explicit platform and estimator selection for statistical
  recomputation and rejected CLI, plan or input-platform mismatches.
- Distinguished the new version 2.0.0 release identity from upstream pre-v2
  commit `d096a2020649d360dfffc860309a510734ea6033`.
- Added reproducible setup, demonstration, test and paper-contract shell
  entry points.

## 1.0.0

- Established the public satellite and UAV implementation lineage represented
  by upstream commit `d096a2020649d360dfffc860309a510734ea6033`.
