"""Recompute the finite-input, prescribed-thinning Cox diagnostic.

Inputs are observed/supplied count pairs, entry times and stored uniform marks.
No distribution for the count pairs is inferred or invented here. The diagnostic
checks the Cox arrival prediction under fixed retention, not the accuracy of
freezing a controller's evolving retention process.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np


def _rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def void_probability(rho, beta_at_one, retention, duration_s, kappa=.60):
    """Full Cox void probability under piecewise-constant retention."""
    rho, beta, p, duration = np.broadcast_arrays(rho, beta_at_one, retention, duration_s)
    if np.any(rho < 0) or np.any(beta < 0) or np.any(duration < 0) or np.any((p < 0) | (p > 1)) or kappa < 0:
        raise ValueError('invalid Cox geometry, retention, window or kappa')
    return np.exp(rho * np.expm1(-kappa * beta * p * duration))


def estimate_kappa(trace_counts, trace_exposures):
    counts, exposure = np.asarray(trace_counts, float), np.asarray(trace_exposures, float)
    if counts.shape != exposure.shape or not np.isfinite(counts).all() or not np.isfinite(exposure).all() or (counts < 0).any() or (exposure < 0).any() or exposure.sum() <= 0:
        raise ValueError('nonnegative matched counts and positive total exposure required')
    return float(counts.sum() / exposure.sum())


def bootstrap_kappa(trace_counts, trace_exposures, seed_ids, *, n_bootstrap=10000, random_seed=20261001):
    """Stratified bootstrap of complete traces within each environment seed."""
    counts, exposure, seeds = np.asarray(trace_counts, float), np.asarray(trace_exposures, float), np.asarray(seed_ids)
    if counts.shape != exposure.shape or counts.shape != seeds.shape:
        raise ValueError('counts, exposures, seed IDs must match')
    estimate_kappa(counts, exposure)
    rng = np.random.default_rng(random_seed)
    count_sum, exposure_sum = np.zeros(n_bootstrap), np.zeros(n_bootstrap)
    for seed in np.unique(seeds):
        idx = np.flatnonzero(seeds == seed)
        draw = rng.choice(idx, size=(n_bootstrap, len(idx)), replace=True)
        count_sum += counts[draw].sum(axis=1)
        exposure_sum += exposure[draw].sum(axis=1)
    if (exposure_sum <= 0).any():
        raise ValueError('bootstrap resample has zero exposure')
    return count_sum / exposure_sum


def recompute_calibration(data_root, out=None):
    """Rebuild all 18,000 labels and five-seed sensitivity summaries from files."""
    root = Path(data_root)
    archive = root / 'archive' if (root / 'archive').is_dir() else root
    source = archive / 'sources_kappa'
    inputs = _rows(archive / 'v19_diag_retention_inputs.csv')
    key = lambda r: (int(r['environment_seed']), r['trace_id'], r['user_id'])
    retention = {key(r): float(r['p_diag']) for r in inputs}
    errors = []
    for row in inputs:
        F, V = int(row['diagnostic_feasible_count']), int(row['diagnostic_visible_count'])
        expected = F / V if V else 0.
        if F < 0 or V < F or not np.isclose(float(row['p_diag']), expected, atol=1e-14, rtol=0):
            errors.append({'kind': 'retention_count_ratio', 'user_id': row['user_id']})
    event_times = defaultdict(list)
    events = _rows(source / 'raw_entry_events.csv.gz')
    mark_checks = 0
    for row in events:
        if row['split'] != 'heldout_test':
            continue
        k = key(row)
        if k not in retention:
            errors.append({'kind': 'missing_retention_input', 'user_id': row['user_id']}); continue
        keep = float(row['thinning_uniform_mark']) < retention[k]
        mark_checks += 1
        if keep != bool(int(row['feasible_entry'])):
            errors.append({'kind': 'stored_retention_flag', 'event_id': row['event_id']})
        if keep:
            event_times[k].append(float(row['relative_time_s']))
    for k, times in event_times.items():
        event_times[k] = np.sort(times)
    windows = _rows(source / 'raw_windows.csv.gz')
    kappas = sorted({float(r['kappa']) for r in _rows(source / 'kappa_sensitivity.csv')})
    strata = defaultdict(list)
    window_rows = []
    for row in windows:
        k = key(row)
        p = retention[k]
        start, duration = float(row['window_start_s']), float(row['window_duration_s'])
        times = event_times.get(k, np.array([]))
        count = int(np.searchsorted(times, start + duration, side='right') - np.searchsorted(times, start, side='right'))
        y = int(count == 0)
        if count != int(row['feasible_entry_count']) or y != int(row['void_label']):
            errors.append({'kind': 'window_label', 'window_id': row['window_id']})
        beta, rho = float(row['beta_at_kappa_1_per_s']), float(row['rho'])
        if not np.isclose(float(row['p_feas_frozen']), p, rtol=0, atol=1e-14):
            errors.append({'kind': 'window_retention', 'window_id': row['window_id']})
        for kappa in kappas:
            p0 = float(void_probability(rho, beta, p, duration, kappa))
            metrics = (p0, y, (p0-y)**2)
            strata[(k[0], kappa, int(duration))].append(metrics)
        window_rows.append({'window_id': row['window_id'], 'environment_seed': k[0],
                            'trace_id': k[1], 'user_id': k[2], 'start_s': start,
                            'duration_s': duration, 'retention': p, 'retained_entries': count, 'void_label': y})
    seed_rows, sensitivity = [], []
    for kappa in kappas:
        units = []
        for seed in sorted({k[0] for k in strata}):
            duration_means = [np.mean(strata[s], axis=0) for s in strata if s[0] == seed and s[1] == kappa]
            mean = np.mean(duration_means, axis=0)
            units.append(mean)
            seed_rows.append({'environment_seed': seed, 'kappa': kappa, 'p0_mean': mean[0],
                              'void_label_mean': mean[1], 'brier_mean': mean[2]})
        units = np.asarray(units)
        row = {'kappa': kappa, 'n_environment_seeds': len(units), 'n_windows': len(windows)}
        for i, metric in enumerate(['p0', 'void_label', 'brier']):
            row[f'{metric}_mean'] = float(units[:, i].mean())
            row[f'{metric}_sample_sd'] = float(units[:, i].std(ddof=1))
        sensitivity.append(row)
    summary_checks = 0
    expected_rows = {float(r['kappa']): r for r in _rows(source / 'kappa_sensitivity.csv')}
    for row in sensitivity:
        for metric in ['p0_mean', 'p0_sample_sd', 'void_label_mean', 'void_label_sample_sd', 'brier_mean', 'brier_sample_sd']:
            summary_checks += 1
            if not np.isclose(row[metric], float(expected_rows[row['kappa']][metric]), atol=1e-12, rtol=1e-10):
                errors.append({'kind': 'sensitivity_summary', 'kappa': row['kappa'], 'metric': metric})
    trace_rows = _rows(source / 'calibration_trace_counts.csv')
    estimate = estimate_kappa([float(r['geometric_entry_count']) for r in trace_rows],
                              [float(r['geometric_exposure_kappa_1']) for r in trace_rows])
    report = {'status': 'passed' if not errors else 'failed', 'n_retention_inputs': len(inputs),
              'n_windows': len(windows), 'n_entry_mark_checks': mark_checks,
              'n_summary_checks': summary_checks, 'kappa_hat': estimate,
              'retention_mean': float(np.mean(list(retention.values()))), 'errors': errors,
              'scope': 'Finite-input prescribed-thinning arrival diagnostic; no closed-loop load calibration.'}
    if out is not None:
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        for filename, rows in [('calibration_windows.csv', window_rows), ('calibration_seed_metrics.csv', seed_rows), ('calibration_sensitivity.csv', sensitivity)]:
            with (out/filename).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (out/'calibration_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
