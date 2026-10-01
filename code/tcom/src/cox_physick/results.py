"""Portable supplied-results archive, independent reduction, and audit.

The unmodified source tables are in ``archive/`` and numerical execution arrays
in ``raw/``. New manifests use relative paths, training-seed terminology for the
author-confirmed learning repetitions, and episode statistics for rule methods.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import batch_prefix_metrics, summarize

CONTROL_METRICS = ['outage_percent', 'hof_percent', 'pp_percent', 'tp_Mbps',
                   'served_p5tp_Mbps', 'peak_normalized', 'active_load_variance',
                   'Cbar', 'Lbar', 'J_user', 'J_network']
LEDGER_METRICS = ['attempt_associated_outage_percent', 'nonattempt_outage_percent',
                  'attempt_percent', 'executed_ho_per_user_min', 'weighted_outage',
                  'weighted_ho', 'weighted_load', 'weighted_negative_rate']
COUNT_FIELDS = ['outage_count', 'served_count', 'ho_attempt_count', 'ho_failure_count',
                'pp_event_count', 'interbeam_exec_ho_count', 'intersat_exec_ho_count',
                'rate_sum_Mbps', 'load_square_sum']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(''); return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_archive_index(data_root):
    root = Path(data_root); archive = root / 'archive'
    cases = read_csv(archive / '01_统一主结果_115条件.csv')
    ledger = {r['case_id']: r for r in read_csv(archive / 'v18_event_cost_ledger_115conditions.csv')}
    bindings = read_csv(archive / 'source_records/03_derived/simulation_bindings.csv')
    episode_index = read_csv(archive / 'v19_load_episode_occupancy_audit.csv')
    manifests, raw_rows, episode_rows, summaries = [], [], [], []
    for c in cases:
        l = ledger[c['case_id']]
        record = {k: c[k] for k in ['case_id', 'method_id', 'paper_name', 'N_sat', 'K', 'H', 'weight_id', 'tau_hyst', 'evaluation_job_id']}
        record.update(aggregation_unit=l['aggregation_unit'], n_training_seeds=c['n_training_seeds'],
                      episodes_per_seed_or_rule=c['episodes_per_seed_or_rule'],
                      raw_files=';'.join('raw/' + Path(p).name for p in c['raw_files'].split(';')),
                      weight_outage=l['weight_outage'], weight_ho=l['weight_ho'], weight_load=l['weight_load'], weight_rate=l['weight_rate'])
        manifests.append(record)
        summary = dict(record)
        for metric in CONTROL_METRICS + ['mse_user_vector', 'persistence_mse_user_vector']:
            for suffix in ['mean', 'sample_sd', 'n_units', 'n_valid_by_unit']:
                summary[f'{metric}_{suffix}'] = c.get(f'{metric}_{suffix}', '')
        for metric in LEDGER_METRICS:
            for suffix in ['mean', 'sample_sd', 'n_units']:
                summary[f'{metric}_{suffix}'] = l[f'{metric}_{suffix}']
        summaries.append(summary)
    for b in bindings:
        case = next(c for c in cases if c['case_id'] in b['case_ids'].split(';'))
        raw_rows.append({'raw_file': 'raw/' + Path(b['raw_file']).name, 'job_id': b['job_id'],
                         'method_id': b['method_id'], 'training_seed': b['simulation_repeat'],
                         'aggregation_unit': ledger[case['case_id']]['aggregation_unit'],
                         'N_sat': b['N_sat'], 'K': b['K'], 'saved_H': b['saved_H'],
                         'requested_H': b['requested_H'], 'case_ids': b['case_ids'],
                         'episodes': b['unique_synthetic_episodes'], 'episode_prefixes': b['episode_prefix_rows'],
                         'association_id_stride': 2, 'empty_association_id': -1,
                         'encoding': 'satellite_id * 2 + stored_beam_id',
                         'array_axes': 'epoch,episode,user'})
    for e in episode_index:
        episode_rows.append({k: e[k] for k in ['case_id', 'training_seed', 'episode_id', 'raw_episode_index_zero_based', 'N_sat', 'K', 'H']} | {'raw_file': 'raw/' + Path(e['raw_file']).name})
    write_csv(root / 'case_manifest.csv', manifests)
    write_csv(root / 'raw_manifest.csv', raw_rows)
    write_csv(root / 'episode_manifest.csv', episode_rows)
    write_csv(root / 'results_summary.csv', summaries)
    index = {'conditions': len(manifests), 'raw_files': len(raw_rows), 'episode_prefixes': len(episode_rows),
             'author_confirmation': 'The author identifies the supplied numerical records as experimental results; inherited source labels are preserved verbatim.',
             'statistics': 'Learning methods: episode means within each training seed, then mean/sample SD of seed means. Rule methods: mean/sample SD across episodes.',
             'association_encoding': 'Archived pair IDs use stride two; this is a storage encoding and does not determine the configured number of beams.',
             'paths': {'summary': 'results_summary.csv', 'cases': 'case_manifest.csv', 'raw': 'raw_manifest.csv', 'episode_prefixes': 'episode_manifest.csv',
                       'calibration_inputs': 'archive/v19_diag_retention_inputs.csv', 'calibration_records': 'archive/sources_kappa'}}
    (root / 'manifest.json').write_text(json.dumps(index, indent=2, ensure_ascii=False) + '\n')
    return index


def _metrics_from_episode_row(r, case):
    K, H = int(case['K']), int(case['H']); n = K * H
    c = {field: float(r[field]) for field in COUNT_FIELDS}
    wo, wh, wl, wr = (float(case['weight_' + name]) for name in ['outage', 'ho', 'load', 'rate'])
    c.update(outage_percent=100*c['outage_count']/n,
             hof_percent=100*c['ho_failure_count']/c['ho_attempt_count'] if c['ho_attempt_count'] else np.nan,
             pp_percent=100*c['pp_event_count']/(K*(H-2)) if H>2 else np.nan,
             tp_Mbps=c['rate_sum_Mbps']/n,
             Cbar=(.1*c['interbeam_exec_ho_count']+.3*c['intersat_exec_ho_count'])/n,
             Lbar=c['load_square_sum']/(10*n),
             attempt_percent=100*c['ho_attempt_count']/n,
             attempt_associated_outage_percent=100*c['ho_failure_count']/n,
             nonattempt_outage_percent=100*(c['outage_count']-c['ho_failure_count'])/n,
             executed_ho_per_user_min=60*(c['interbeam_exec_ho_count']+c['intersat_exec_ho_count'])/n)
    for metric in ['served_p5tp_Mbps', 'peak_normalized', 'active_load_variance']:
        c[metric] = float(r[metric]) if r[metric] not in ('', 'NA') else np.nan
    c.update(weighted_outage=wo*c['outage_percent']/100, weighted_ho=wh*c['Cbar'],
             weighted_load=wl*c['Lbar'], weighted_negative_rate=-wr*c['tp_Mbps']/240)
    c['J_user'] = sum(c[k] for k in ['weighted_outage', 'weighted_ho', 'weighted_load', 'weighted_negative_rate'])
    c['J_network'] = K*c['J_user']
    return c


def audit_results(data_root, out=None, full=False):
    """Check summaries; with full=True independently reduce all execution NPZs.

    Writes per-episode reductions, all condition summaries, request-state outage
    decompositions and a concise discrepancy report. Does not inspect checkpoint
    selection, infer physical causes, or change the supplied result values.
    """
    root = Path(data_root)
    if not (root/'manifest.json').exists():
        build_archive_index(root)
    archive = root/'archive'
    cases = {r['case_id']: r for r in read_csv(root/'case_manifest.csv')}
    expected = {r['case_id']: r for r in read_csv(root/'results_summary.csv')}
    episode_index = read_csv(root/'episode_manifest.csv')
    source_episodes = {(r['case_id'],r['training_seed'],r['episode_id']): r for r in read_csv(archive/'source_records/02_to_fill/episode_results.csv')}
    errors, error_count, checks = [], 0, defaultdict(int)
    def check(ok, kind, **context):
        nonlocal error_count
        checks[kind] += 1
        if not ok:
            error_count += 1
            if len(errors) < 100:
                errors.append({'kind': kind, **context})
    recomputed = []
    if full:
        by_raw = defaultdict(list)
        for entry in episode_index:
            by_raw[entry['raw_file']].append(entry)
        for file_index, (relative, entries) in enumerate(by_raw.items(), 1):
            path = root/relative
            with np.load(path, allow_pickle=False) as record:
                case = cases[entries[0]['case_id']]
                try:
                    values = batch_prefix_metrics(record['association'], record['proposal'], record['rate_Mbps'],
                        horizons=sorted({int(e['H']) for e in entries}), capacity=10, beta_load=.8,
                        weights=tuple(float(case['weight_'+name]) for name in ['outage','ho','load','rate']),
                        n_sat=int(case['N_sat']), beams_per_satellite=2)
                except (ValueError, AssertionError) as error:
                    check(False, 'raw_structure', raw_file=relative, error=str(error)); continue
                checks['raw_files_reduced'] += 1
                for entry in entries:
                    h, e = int(entry['H']), int(entry['raw_episode_index_zero_based'])
                    v = {name: a[e].item() for name, a in values[h].items()}
                    src = source_episodes[(entry['case_id'],entry['training_seed'],entry['episode_id'])]
                    for field in COUNT_FIELDS + ['served_p5tp_Mbps','peak_normalized','active_load_variance']:
                        target = float(src[field]) if src[field] not in ('','NA') else np.nan
                        ok = np.isclose(v[field], target, rtol=1e-10, atol=1e-10, equal_nan=True)
                        check(ok, 'episode_field', case_id=entry['case_id'], training_seed=entry['training_seed'], episode_id=entry['episode_id'], field=field, actual=v[field], expected=target)
                    decomposition = sum(v[k] for k in ['source_retention_outage_count','empty_source_access_outage_count','nonempty_source_empty_request_outage_count','empty_source_empty_request_outage_count'])
                    check(decomposition == v['nonattempt_outage_count'], 'outage_partition', case_id=entry['case_id'], episode_id=entry['episode_id'])
                    recomputed.append(dict(entry) | v)
    else:
        for entry in episode_index:
            source = source_episodes[(entry['case_id'],entry['training_seed'],entry['episode_id'])]
            recomputed.append(dict(entry) | _metrics_from_episode_row(source, cases[entry['case_id']]))
    grouped = defaultdict(list)
    for row in recomputed:
        grouped[row['case_id']].append(row)
    summaries, decompositions = [], []
    for case_id, case in cases.items():
        rows = grouped[case_id]
        check(len(rows) == len([e for e in episode_index if e['case_id'] == case_id]), 'episode_coverage', case_id=case_id)
        if not rows:
            continue
        seeds = [r['training_seed'] for r in rows] if case['aggregation_unit'] == 'training_seed_mean' else None
        summary = {'case_id':case_id, 'aggregation_unit':case['aggregation_unit']}
        for metric in CONTROL_METRICS + LEDGER_METRICS:
            stats = summarize([r[metric] for r in rows], seeds)
            for stat in ['mean', 'sample_sd', 'n_units']:
                key = f'{metric}_{stat}'; summary[key] = stats[stat]
                target = expected[case_id].get(key, '')
                if target:
                    check(stats[stat] is not None and np.isclose(stats[stat], float(target), rtol=1e-9, atol=1e-9), 'condition_summary', case_id=case_id, field=key, actual=stats[stat], expected=float(target))
        summaries.append(summary)
        if full:
            fields = ['outage_count','ho_attempt_count','ho_failure_count','nonattempt_outage_count','source_retention_outage_count','empty_source_access_outage_count','nonempty_source_empty_request_outage_count','empty_source_empty_request_outage_count','executed_ho_count']
            totals = {field: int(sum(r[field] for r in rows)) for field in fields}
            totals.update(case_id=case_id, total_user_epochs=len(rows)*int(case['K'])*int(case['H']))
            decompositions.append(totals)
    report = {'status':'passed' if not error_count else 'failed', 'mode':'raw_array_reduction' if full else 'episode_table_reduction',
              'conditions':len(cases), 'episode_prefixes':len(recomputed), 'checks':dict(checks),
              'discrepancy_count':error_count, 'discrepancies':errors,
              'scope':'Numerical execution metrics, aggregation, occupancy and event accounting; geometric visibility and historical training are outside this reduction.'}
    if out is not None:
        out = Path(out); out.mkdir(parents=True,exist_ok=True)
        write_csv(out/'recomputed_episode_metrics.csv',recomputed)
        write_csv(out/'recomputed_condition_summaries.csv',summaries)
        if decompositions:
            write_csv(out/'recomputed_outage_partitions.csv',decompositions)
        (out/'results_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
