# Physical execution environment

`configs/paper.yaml` specifies the physical simulation and controller defaults implemented by this package. Evaluation outputs record the effective configuration, physical traces, requests, and executed associations.

## API

```python
from cox_physick.config import Config
from cox_physick.environment import Environment

cfg = Config.from_yaml("configs/paper.yaml")
env = Environment(cfg, seed=2026, episode_id=0)  # exogenous seed, not training seed
obs = env.reset()
# controller selects one feasible candidate slot per user, or -1 when none exists
result = env.step(request_slots)
obs = result.next_observation  # None after the configured horizon
env.export_exogenous("outputs/episode_exogenous.npz")
```

`Config.environment`, `.model`, and `.training` are dictionaries. `.weights` is the tuple `(w_o,w_h,w_l,w_r)`. `default_config()` supplies the full defaults; `Config.to_yaml(path)` writes them. All geometry, channel powers, costs, and statistics use float64. Neural inputs can be converted by the learning code to float32.

An `Observation` contains padded `[K,8]` satellite/beam IDs, decision feasibility, candidate subband, raw linear SINR, physical RST, analytic utility, post-join load, and the `[K,8,6]` edge descriptor. IDs `-1` indicate padding; `exists` and `current` are derived boolean properties. The common Cox window is the previous satellite's RST while visible, otherwise one second. `survival` is the candidate-window Cox void probability, not a handover-failure probability. `z` uses the paper's log transforms only on normalized entry rate and RST; SINR remains a linear ratio and load is the delayed broadcast fraction.

`StepResult` contains rewards, costs and their executable components, final associations/subbands/rates, satellite occupancy, attempt/failure indicators, and `info`. An attempt requires a nonempty source, a nonempty request, and a changed satellite/beam pair. `info` preserves candidate measurements, tentative assignments, capacity rejection, the simultaneous check and final SINRs, same-boresight S/I/N, executed handovers, and the disjoint non-attempt outage request states. Empty-source access is separately split by whether the user has ever been served.

## Orbit, coordinates, and visibility

- Synthetic Walker shell: 550 km, 53 degrees, 20 equally spaced planes, `F=1`; mean anomaly `2π(j/Q+F p/N)` and ascending node `2πp/P`.
- SGP4 initialization uses WGS72, improved mode `i`, eccentricity/argument of perigee/B*/first and second mean-motion derivatives all zero. Mean motion is computed from WGS72's gravitational constant and the nominal 6921 km semi-major axis.
- `start_utc` is explicitly a new-run reference epoch (`2026-09-27T00:00:00Z` by default). Episode `e` begins `e*episode_spacing_s` later, propagating the same initialized constellation rather than resetting the shell.
- TEME positions rotate to PEF using Vallado GMST at UT1, then to ECEF using the configured polar motion. Default `UT1−UTC=0` and `xp=yp=0` are explicit Earth-orientation approximations; no external IERS table is silently downloaded.
- Users are uniform in longitude and sine latitude over 30–45 degrees N, −10–10 degrees E, stationary throughout an episode. A frozen pool of 500 positions supplies nested user sets for density comparisons and external beam planning.
- Visibility uses elevation at least 25 degrees. RST is the first future one-second sample below this threshold, obtained from propagated geometry. Propagation extends 1200 seconds beyond the episode. Insufficient extension raises an error rather than clipping RST.

In ephemeris mode (`environment.descriptor_mode: ephemeris`), exact one-second geometric entry events are computed once and stored sparsely. Currently visible satellites are excluded from each future-entry count. `eph_z` replaces common-window `P0` and entry rate by `(1-p)^n` and `pn/T`, and `eph_survival` applies the same frozen `p` to each candidate's own window. No Cox κ multiplier is applied to these known entries.

## Exogenous beams and weather

Every 20 seconds, each satellite considers the geometrically visible planning points. A target covers other eligible points within one degree as seen from that satellite. Greedy maximum uncovered count selects at most seven targets, breaking ties by point ID; the first four are enabled. The one-degree planning radius is not a hard service cutoff. Physical antenna gain and SINR determine service.

New and old targets are matched by minimum total angular distance at the current satellite position. Equal optimal matchings (tolerance `1e-12` radians) choose ascending beam ID for ascending target ID; unmatched targets receive ascending unused IDs. Targets remain Earth fixed for 20 seconds while pointing vectors are recomputed every second. The planner never reads association, controller scores, loads, or future weather.

Rain defaults to independent Bernoulli(0.10) per planning user, constant for the episode and shared across all paths received by that user. Blockage is an independent per-user two-state chain, initially stationary, with per-second clear→blocked 0.005263 and blocked→clear 0.1. Its state is likewise common to desired and interfering paths to that user. Fixed losses are gas 0.12 dB, rainy 5 dB, blocked 8 dB. `rain_correlation` and `blockage_correlation` may explicitly select `user`, `region`, or `link`; changing them defines a different environment. Sampling the full fixed pool maintains identical weather for the same user IDs across density settings.

## Candidate measurements

All requests use one frozen previous-execution snapshot. Transmissions whose beam is disabled in the current plan are cleared from this measurement snapshot; no current request or admission is inserted. Delayed occupancy remains the preceding executed satellite load.

For a candidate on the user's previous satellite, the measurement retains its previous band. Otherwise it uses the smallest currently free band in the frozen snapshot. If all bands are occupied, band zero is used solely as a reference measurement; post-join capacity feasibility still excludes a full target. `candidate_subband_reason` records `retained`, `lowest_free`, `full_satellite_reference`, or `padded`.

The evaluated user's old transmission is excluded from its alternative candidate measurement. Receive boresight points at that candidate satellite; every interferer in that same SINR is evaluated relative to the same boresight. This is a declared hypothetical-release measurement, shared across controllers. Candidate and final assigned subbands need not coincide: concurrent admissions can consume the measured free band before a user is admitted.

Visible enabled pairs are ranked by decreasing frozen SINR, then satellite/beam ID; at most eight become graph edges. A feasible source replaces the last preselected pair when needed. Decision feasibility is the intersection with SINR ≥ 0 dB and post-join load ≤ 10. The retention fraction counts unique feasible satellites over **all** visible satellites, before top-eight selection.

## Joint admission and execution

1. Reserve source slots/bands for all users whose current serving pair is in the decision set. Keep reservations through the epoch, even when a user is admitted elsewhere.
2. Existing users requesting another feasible beam on the same satellite keep their prior band with priority and consume no additional slot. This also applies when their old beam is infeasible; such a request is recorded by `same_satellite_priority`.
3. Rank remaining external requests by decreasing frozen target SINR, breaking ties by user ID. Assign the smallest free band within residual satellite capacity. A capacity-rejected request falls back only to a previously reserved source; otherwise it has no tentative link. An inactive source reservation consumes capacity but transmits no interference.
4. Freeze one tentative transmission per user and check visibility and physical SINR simultaneously. Remove all failing transmissions together. There is one check, no second admission, no subband reallocation and no second source fallback.
5. Recompute interference and rates for surviving transmissions on unchanged bands and boresights. Removing interferers cannot reduce survivor SINR. Rates are `20 log2(1+SINR)` Mbps; nonserved rates are zero.

Subbands are orthogonal across a satellite's beams and fully reused between satellites. A physical link uses 10 dBW per active subband, 12 GHz, 20 MHz bandwidth, and −96 dBm noise. Tx/Rx peaks are 40/35 dBi; full 3 dB beamwidths are 2/3 degrees; attenuation caps are 30/40 dB. Earth-occulted paths (negative geometric elevation) contribute zero desired or interfering power; the 25-degree service threshold is a separate condition. Both patterns use `Gmax − min(12(ψ/θ3dB)^2,Amax)`. Desired and interference powers include the same tagged receive orientation and the receiving user's weather state.

## Records and verification

`export_exogenous` saves satellite ECEF geometry (including the RST extension), exact UTC start, user positions, planning pool, observed weather states, and all beam target/enable records. Per-epoch `info` supplies the remaining physical and event diagnostics. These outputs permit checking geometric visibility, resource uniqueness, link budgets, attempt denominators, one-pass execution, and final costs without recovering state from averages.

The per-user cost is `wo*outage + wh*Cexec + wl*executed_load/capacity − wr*rate/Rref`. The full-population mean load contribution equals `sum_sat occupancy²/(capacity*K)`, including zero service during outage. Tests cover this identity, deterministic allocation and fallback, the invalid-old-beam same-satellite boundary, one-pass failures, shared receive pointing, nested exogenous inputs, exact Cox boundaries, and coordinate rotation.

Run `python -m pytest tests/test_environment.py`.

## Propagation references

The SGP4 initialization and TEME coordinate convention follow the [maintained SGP4 package documentation](https://pypi.org/project/sgp4/). TEME-to-PEF uses the [Vallado implementation](https://celestrak.org/publications/AIAA/2006-6753/); Earth-orientation defaults are recorded above.
