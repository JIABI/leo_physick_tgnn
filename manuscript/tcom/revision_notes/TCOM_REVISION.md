# v22 revision and implementation notes

## Manuscript edits

Only the following prose and layout changes were needed beyond v21:

1. The nominal outage partition explicitly states that no outage with an empty
   request was recorded for either Cox-PhysiCK or greedy. This comes from the
   supplied requests and executed associations; it is not a restriction on
   states that the controller can encounter.
2. Appendix B-E points to the accompanying archive containing all 3,000
   trace–user diagnostic count pairs and their window mappings. The actual
   event times and uniform marks are included, so the 18,000 labels and
   calibration curves can be recomputed directly.
3. The conclusion is kept together after Fig. 3. Its claims are unchanged.

Cox/System Model/PhysiCK equations, all reported result rows, references,
original figures and the checkpoint discussion are preserved. The separate
code-generated plots use the same recorded coordinates and are supplied as
editable reproduction outputs; they do not replace the manuscript's figures.

## What the records resolve

The archive audit reconstructs satellite occupancy from executed associations,
then recomputes outage, attempts, conditional HOF, ping-pong events, throughput,
Cbar, Lbar, Peak and average cost. It checks the outage partition, capacity and
load-square identities before aggregation. Source records and recomputed
condition/episode tables are both supplied. No new mean or SD was fitted to an
expected ranking.

The DRL comparator remains the **adapted parameter-sharing hysteretic DQN**:
shared per-user Q learning, replay, target network, feasible-action masking and
asymmetric TD-loss weighting. The implementation does not label these mechanisms
as the complete successive algorithm of the cited paper. Comparisons concern
this common-environment adaptation.

## Executable resource rules

`code/tcom/configs/tcom.yaml` and `docs/TCOM_ENVIRONMENT.md` now fully specify and
implement candidate-band mapping, subband allocation, joint execution,
beam-target planning, receiver pointing and weather correlation. These close
the software interface gaps and produce detailed physical records for new runs.
The supplementary rule-reference CSV lists these choices as prospective rules;
the stored association/rate arrays do not independently identify which choices
were used historically. Accordingly, the old result archive is preserved and
the new implementation has a separate run/output entry point. It is not
claimed that writing the implementation reran the historical experiments.

The question of historical resource-rule equivalence remains for the author;
no checkpoint investigation is added in this revision. If those rules differ,
new physical runs can be produced through the supplied experiment plan without
changing the existing archive.

## Code lineage

Upstream repository: https://github.com/JIABI/leo_physick_tgnn.git,
commit `9564248`. Its current root and satellite/UAV configuration concern a
separate controller-facing-state study. They are retained. TCOM therefore has
its own package, CLI, configuration, data index and statistical estimator,
rather than silently running the other study's link proxy or 0.1 s control grid.

The new implementation has physical SGP4 propagation and link-budget execution,
not a lookup of the manuscript's outcome table. Recorded-result reproduction
is a distinct calculation from new training and simulation.
