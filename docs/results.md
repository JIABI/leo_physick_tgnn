# Final results and data conventions

`data/results_summary.csv` is the authoritative summary for the evaluation settings used by the manuscript. `results/tables/` provides paper-oriented views of that same table. Files use experiment names and conditions rather than revision identifiers.

## Execution data

The data index has three levels:

- `case_manifest.csv`: method, constellation size, users, evaluation prefix, objective weights, hysteresis, training setting, and input file paths.
- `raw_manifest.csv`: one entry per execution file, including its training seed and association encoding.
- `episode_manifest.csv`: one entry per case, training seed, and evaluated episode prefix.

Paths in these manifests are relative to `data/`. The files in `raw/` contain `association`, `proposal`, and `rate_Mbps` with axes `[epoch, episode, user]`. An empty association is `-1`. Their stored pair IDs use `satellite_id * 2 + stored_beam_id`, as declared in the manifest. This is a storage encoding; the physical simulator has seven beam identities per satellite and records its encoding separately.

The dataset contains final H=200 and H=800 evaluations. Different prefixes reuse the same saved episode, rather than representing independent experiments. Hysteresis settings reuse the corresponding trained model, while the 17 distinct cost-weight configurations have separate training seeds.

## Metrics

| Metric | Definition |
|---|---|
| Outage (%) | `100 * outage_count / (K*H)` |
| Conditional HOF (%) | `100 * failed_attempts / attempts` |
| PP (%) | Nonempty executed A–B–A events divided by `K*(H-2)`, multiplied by 100 |
| TP (Mbps) | Executed rate sum divided by `K*H`, including zeros during outage |
| Served P5TP (Mbps) | Fifth percentile of served-link rates within each episode |
| Peak | Prefix maximum of normalized occupancy smoothed with factor 0.8 from zero |
| `Cbar` | `(0.1 * interbeam_HO + 0.3 * intersatellite_HO) / (K*H)` |
| `Lbar` | `sum(occupancy**2) / (10*K*H)` over satellites and epochs |
| `J_user` | `w_o * outage_fraction + w_h * Cbar + w_l * Lbar - w_r * TP/240` |
| `J_network` | `K * J_user` |
| Attempt rate (%) | Attempts divided by all user–epochs, multiplied by 100 |
| Executed HO frequency | Nonempty-to-different-nonempty execution changes per user per minute |

An attempt requires a nonempty source and a different nonempty requested target. A failed attempt ends in outage. Initial access does not count as handover, and a rejected request followed by source retention is not an outage failure. HOF is missing when no attempts occur; P5TP is missing when no samples are served.

The dataset retains executed load variance as a supporting diagnostic. It is not a replacement for the cost's `Lbar` or the paper's Peak statistic.

## Statistical units

For learned methods, valid episode metrics are averaged within each of five training seeds. The reported mean and sample SD are then calculated across those five means. Deterministic methods use the mean and sample SD across the 30 shared test episodes. These SDs describe different sources of variation.

Pooled event rates are labeled separately from the mean of episode-level rates. In particular, pooling HOF numerators and denominators need not equal averaging per-episode conditional HOF. Cost intervals relative to greedy condition on its fixed test-set mean and use the five learned-seed cost differences. Learned-versus-learned comparisons are descriptive.

## Calibration diagnostic

`calibration/retention_inputs.csv` supplies 3,000 trace–user count pairs and their fixed retention ratios. The entry events, uniform thinning marks, trace exposures, and six nested window durations are stored alongside them. The diagnostic has 18,000 windows and uses complete-trace resampling within environment seeds for calibration uncertainty.

This is the paper's prescribed frozen-retention diagnostic. It tests the arrival prediction under those inputs, not the accuracy of freezing an evolving controller-generated retention process. The controller's feasibility estimator is separately defined by the physical observation and load rules.

## Recompute and verify

```python
from cox_physick.results import audit_results
from cox_physick.calibration import recompute_calibration
from cox_physick.publication import export_publication_results

audit_results("data", "outputs/audit", full=True)
recompute_calibration("data", "outputs/calibration")
export_publication_results("data", "outputs/tables")
```

`full=True` derives metrics from execution arrays. The default audit uses the episode table. Both compare results against the same authoritative summary. The recorded execution arrays support metric, load, event, and aggregation checks; they do not contain all of the physical geometry, resource allocations, and weather fields produced by a new simulator run.
