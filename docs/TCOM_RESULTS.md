# TCOM result archive

`data/tcom/` is the portable data package for the current Cox-PhysiCK revision. All supplied numerical results are retained. The author confirmed that these are experimental values and that some inherited filenames and metadata were not updated. Those original files remain unchanged in `archive/`; the entry tables below use the manuscript's current terminology.

## Start here

| File | Contents |
|---|---|
| `results_summary.csv` | 115 conditions, original means and sample SD, conditions and aggregation units |
| `case_manifest.csv` | 115 condition definitions, objective weights and relative raw-file paths |
| `raw_manifest.csv` | 315 execution files, training seed, represented prefixes and pair-ID encoding |
| `episode_manifest.csv` | 12,690 episode-prefix records with raw file and zero-based episode index |
| `raw/` | All 315 supplied NPZ execution records, copied without modification |
| `archive/` | Complete existing data collection, including earlier result views, diagnostic inputs, source tables and selection records |
| `checks_full/` | Independent execution-array reductions and their comparison with reported values |
| `checks_calibration/` | Recomputed calibration labels and sensitivity summaries |

Paths in the new manifests are relative to `data/tcom/`. Historical source-location strings remain in the unmodified archive for traceability. They are not required by the new reducers. Existing selection records are preserved; checkpoint selection is outside this revision's audit.

## What the raw arrays contain

The main arrays are `association`, `proposal`, and `rate_Mbps`, with axes `[epoch, episode, user]`. Empty association is `-1`. Archived association IDs encode `satellite_id * 2 + stored_beam_id`. This two-slot storage encoding is declared in `raw_manifest.csv`; it is distinct from the seven-beam configuration of the executable environment. The metrics API accepts its encoding stride explicitly.

The arrays also contain saved candidate/label/prediction diagnostics. The historical diagnostic MSE tables are retained in the archive, rather than promoted to a new primary control result. Execution records alone do not contain the geometry, antenna orientations, subband allocations, or same-link signal/interference/noise decomposition needed to replay the original physical links.

## Metrics and aggregation

All user-epoch means include outage epochs in their denominator. A handover attempt requires both a nonempty previous association and a nonempty, different request. A failed attempt has an empty executed association. Initial access and an empty request are not counted as handover attempts.

| Quantity | Unit and reduction |
|---|---|
| Outage | `100 * outage_count / (K*H)` |
| Conditional HOF | `100 * failed_attempts / attempts`; undefined when attempts are zero |
| Ping-pong | Nonempty executed A–B–A triples divided by `K*(H-2)`, reported as percent |
| TP | Sum of executed rates in Mbps divided by `K*H`, including zeros |
| Served P5TP | Episode-level fifth percentile of nonempty executed-link rates |
| `Cbar` | `(0.1 * interbeam_HO + 0.3 * intersatellite_HO)/(K*H)` |
| `Lbar` | Executed satellite occupancy squared, summed over epochs/satellites and divided by `10*K*H` |
| Peak | Maximum over the evaluated prefix of normalized executed occupancy smoothed with beta 0.8 and zero initialization |
| `J_user` | `w_o*p_out + w_h*Cbar + w_l*Lbar - w_r*TP/240` |
| `J_network` | `K * J_user` |
| HO frequency | Executed association changes per user per minute, excluding empty-source access |

Learning-method summaries first average valid episodes within each training seed, then report the mean and sample SD across those seed means. Rule-method summaries report the mean and sample SD across valid episodes. These are different sources of variability. Sample SD uses denominator `n-1`; undefined values remain missing, not zero. Pooled HOF counts may differ slightly from the mean of episode-level HOF values.

The independent full reduction checks all 315 execution records, 12,690 episode-prefixes, 152,280 episode fields, 12,690 exhaustive non-attempt-outage partitions, and 6,555 condition summary values. The delivered report contains zero discrepancies. It also validates executed capacity and the occupancy Cauchy inequality directly from associations.

## Calibration is a finite-input diagnostic

`archive/v19_diag_retention_inputs.csv` supplies all 3,000 held-out trace–user inputs. Each has the visible/feasible count pair, its ratio, the trace identifier, user identifier and shared window start. No unrecorded sampling distribution for these ratios is required. Each ratio is reused over the six window durations 10, 30, 60, 120, 240 and 480 seconds.

`archive/sources_kappa/` contains the entry times, stored uniform thinning marks, geometric exposure, windows, predictions and trace-level calibration counts. `calibration.recompute_calibration` reconstructs retained entry counts and all 18,000 void labels from these files. It then reconstructs the five-seed, equal-duration macro summaries. The delivered check verifies 72,427 held-out entry marks and 30 sensitivity mean/SD values, with zero discrepancies. The pooled geometric estimate is 0.6033009313.

This diagnostic evaluates the arrival prediction under the supplied fixed-retention inputs. It does not estimate the error of freezing a time-varying closed-loop retention process. Physical criteria for the original count pairs are not inferred from their values.

## Recompute

After installing the TCOM package, the public Python functions are:

```python
from cox_physick.results import audit_results
from cox_physick.calibration import recompute_calibration

audit_results("data/tcom", "outputs/audit", full=True)
recompute_calibration("data/tcom", "outputs/calibration")
```

`full=False` checks the same statistical summaries from supplied episode tables. `full=True` independently reconstructs metrics from the execution arrays. Both paths preserve the original reported tables.

Fresh simulator outputs and the supplied result archive have separate manifests. Running the new implementation produces new run records; it does not replace the archive or imply that newly specified execution rules were used historically.
