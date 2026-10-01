"""Execution-based TCOM metrics and seed-first statistical aggregation.

Associations are nonnegative integer pair IDs ``satellite * stride + beam``;
``-1`` denotes no association. New simulations use stride seven. The supplied
result records declare their pair-ID stride in the raw manifest.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Mapping
import numpy as np


def batch_prefix_metrics(association, requests, rates_mbps, horizons=None, *,
                         previous=None, capacity=10, beta_load=.8, dt_s=1.,
                         weights=(5., .5, .5, 1.), rate_reference=240.,
                         n_sat=None, beams_per_satellite=7,
                         c_interbeam=.1, c_intersatellite=.3):
    """Reduce H×E×K records to per-episode vectors at each requested prefix.

    HOF is undefined (NaN) for no attempts; P5TP is undefined for no service.
    Empty-source access is not a handover attempt. The load statistic includes
    outage samples in its denominator. Peak is the prefix maximum of a causal
    exponentially smoothed executed load, initialized at zero.
    """
    a, q, rate = np.asarray(association), np.asarray(requests), np.asarray(rates_mbps)
    if a.ndim != 3 or a.shape != q.shape or a.shape != rate.shape:
        raise ValueError('association, requests and rates must share shape H×E×K')
    if not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(q.dtype, np.integer):
        raise ValueError('association and request IDs must be integers')
    if (a < -1).any() or (q < -1).any() or not np.isfinite(rate).all() or (rate < 0).any():
        raise ValueError('invalid IDs or nonfinite/negative rates')
    if capacity <= 0 or dt_s <= 0 or rate_reference <= 0 or not 0 <= beta_load < 1 or beams_per_satellite < 1:
        raise ValueError('invalid capacity, duration, normalization or smoothing')
    if c_interbeam < 0 or c_intersatellite < 0:
        raise ValueError('switching costs must be nonnegative')
    H, E, K = a.shape
    if min(H, E, K) < 1:
        raise ValueError('empty execution array')
    horizons = sorted(set([H] if horizons is None else map(int, horizons)))
    if not horizons or min(horizons) < 1 or max(horizons) > H:
        raise ValueError('prefix must lie within the execution record')
    served = a >= 0
    if (rate[~served] != 0).any():
        raise ValueError('outage samples must have zero rate')
    prev = np.empty_like(a)
    if previous is None:
        prev[0] = -1
        prev[1:] = a[:-1]
    else:
        previous = np.asarray(previous)
        if previous.shape == (E, K):
            prev[0], prev[1:] = previous, a[:-1]
        elif previous.shape == a.shape:
            prev[:] = previous
        else:
            raise ValueError('previous must be initial E×K or full H×E×K')
    attempt = (prev >= 0) & (q >= 0) & (prev != q)
    changed = (prev >= 0) & served & (prev != a)
    intersat = changed & (prev // beams_per_satellite != a // beams_per_satellite)
    pp = np.zeros_like(served)
    pp[2:] = served[2:] & served[1:-1] & served[:-2] & (a[2:] == a[:-2]) & (a[2:] != a[1:-1])
    satellite = a // beams_per_satellite
    # Allocate only IDs actually represented; the declared constellation bound
    # remains a separate validation and is never inferred from a pool annotation.
    M = max(1, int(satellite.max()) + 1)
    if n_sat is not None and ((a[served] // beams_per_satellite >= n_sat).any() or
                              (q[q >= 0] // beams_per_satellite >= n_sat).any()):
        raise ValueError('pair ID outside declared constellation')
    encoded = (np.maximum(satellite, 0) + np.arange(H * E).reshape(H, E, 1) * M)[served]
    occupancy = np.bincount(encoded, minlength=H * E * M).reshape(H, E, M)
    if (occupancy > capacity).any():
        raise ValueError('executed occupancy exceeds satellite capacity')
    active = (occupancy > 0).sum(axis=2)
    sum_load = occupancy.sum(axis=2)
    sum_sq = np.square(occupancy).sum(axis=2)
    slack = active * sum_sq - sum_load ** 2
    if (slack < 0).any():
        raise AssertionError('occupancy Cauchy inequality violated')
    nonattempt = ~served & ~attempt
    counters = {
        'outage_count': (~served).sum(axis=2), 'served_count': served.sum(axis=2),
        'ho_attempt_count': attempt.sum(axis=2), 'ho_failure_count': (attempt & ~served).sum(axis=2),
        'pp_event_count': pp.sum(axis=2), 'interbeam_exec_ho_count': (changed & ~intersat).sum(axis=2),
        'intersat_exec_ho_count': intersat.sum(axis=2), 'rate_sum_Mbps': rate.astype(np.float64).sum(axis=2),
        'load_square_sum': sum_sq, 'active_count_sum': active,
        'nonattempt_outage_count': nonattempt.sum(axis=2),
        'source_retention_outage_count': (nonattempt & (prev >= 0) & (q == prev)).sum(axis=2),
        'empty_source_access_outage_count': (nonattempt & (prev < 0) & (q >= 0)).sum(axis=2),
        'nonempty_source_empty_request_outage_count': (nonattempt & (prev >= 0) & (q < 0)).sum(axis=2),
        'empty_source_empty_request_outage_count': (nonattempt & (prev < 0) & (q < 0)).sum(axis=2),
    }
    safe_active = np.maximum(active, 1)
    variance = np.maximum(0., sum_sq / (capacity ** 2 * safe_active) -
                          np.square(sum_load / (capacity * safe_active)))
    ema, peak, peaks = np.zeros((E, M)), np.zeros(E), {}
    for t in range(max(horizons)):
        ema *= beta_load
        ema += (1 - beta_load) * occupancy[t] / capacity
        peak = np.maximum(peak, ema.max(axis=1))
        if t + 1 in horizons:
            peaks[t + 1] = peak.copy()
    result = {}
    for h in horizons:
        n = K * h
        r = {key: value[:h].sum(axis=0) for key, value in counters.items()}
        r['outage_percent'] = 100 * r['outage_count'] / n
        r['hof_percent'] = np.divide(100. * r['ho_failure_count'], r['ho_attempt_count'],
                                     out=np.full(E, np.nan), where=r['ho_attempt_count'] > 0)
        r['pp_percent'] = 100 * r['pp_event_count'] / (K * (h - 2)) if h > 2 else np.full(E, np.nan)
        r['attempt_percent'] = 100 * r['ho_attempt_count'] / n
        r['attempt_associated_outage_percent'] = 100 * r['ho_failure_count'] / n
        r['nonattempt_outage_percent'] = 100 * r['nonattempt_outage_count'] / n
        r['executed_ho_count'] = r['interbeam_exec_ho_count'] + r['intersat_exec_ho_count']
        r['executed_ho_per_user_min'] = 60 * r['executed_ho_count'] / (n * dt_s)
        r['Cbar'] = (c_interbeam * r['interbeam_exec_ho_count'] + c_intersatellite * r['intersat_exec_ho_count']) / n
        r['Lbar'] = r['load_square_sum'] / (capacity * n)
        r['tp_Mbps'] = r['rate_sum_Mbps'] / n
        r['served_p5tp_Mbps'] = np.array([np.quantile(rate[:h, e][served[:h, e]], .05, method='linear')
                                         if served[:h, e].any() else np.nan for e in range(E)])
        r['peak_normalized'] = peaks[h]
        r['active_load_variance'] = variance[:h].mean(axis=0)
        r['actual_nonempty_satellites_mean'] = active[:h].mean(axis=0)
        r['actual_nonempty_satellites_min'] = active[:h].min(axis=0)
        r['actual_nonempty_satellites_max'] = active[:h].max(axis=0)
        r['minimum_epoch_cauchy_slack'] = slack[:h].min(axis=0)
        wo, wh, wl, wr = weights
        r['weighted_outage'] = wo * r['outage_percent'] / 100
        r['weighted_ho'] = wh * r['Cbar']
        r['weighted_load'] = wl * r['Lbar']
        r['weighted_negative_rate'] = -wr * r['tp_Mbps'] / rate_reference
        r['J_user'] = r['weighted_outage'] + r['weighted_ho'] + r['weighted_load'] + r['weighted_negative_rate']
        r['J_network'] = K * r['J_user']
        result[h] = r
    return result


def episode_metrics(association, requests, rates_mbps, *, previous=None, **kwargs):
    """Scalar metrics for a single H×K episode (same conventions as above)."""
    a, q, rate = map(np.asarray, (association, requests, rates_mbps))
    if a.ndim != 2:
        raise ValueError('single-episode arrays must have shape H×K')
    if previous is not None:
        previous = np.asarray(previous)
        previous = previous[None, :] if previous.ndim == 1 else previous[:, None, :]
    vectors = batch_prefix_metrics(a[:, None, :], q[:, None, :], rate[:, None, :],
                                   previous=previous, **kwargs)[a.shape[0]]
    return {key: value[0].item() for key, value in vectors.items()}


def summarize(values: Iterable[float], seed_ids=None):
    """Mean/sample SD of valid episodes or valid seed means, never pooled seeds."""
    values = np.asarray(list(values), dtype=float)
    if seed_ids is None:
        valid = np.isfinite(values)
        units, n_valid = values[valid], [int(valid.sum())]
    else:
        seed_ids = list(seed_ids)
        if len(seed_ids) != len(values):
            raise ValueError('one seed ID required per episode value')
        grouped = defaultdict(list)
        for seed, value in zip(seed_ids, values):
            grouped[seed].append(value)
        units, n_valid = [], []
        for seed in sorted(grouped, key=str):
            v = np.asarray(grouped[seed]); v = v[np.isfinite(v)]
            n_valid.append(int(v.size))
            if v.size:
                units.append(float(v.mean()))
        units = np.asarray(units)
    return {'mean': float(units.mean()) if len(units) else None,
            'sample_sd': float(units.std(ddof=1)) if len(units) > 1 else None,
            'n_units': len(units), 'n_valid_by_unit': n_valid}
