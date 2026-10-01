"""Final reported results, independent metric reduction, and consistency checks.

All inputs are resolved relative to the data directory. Learning methods use
training-seed means; deterministic controllers use episode statistics.
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
    cases = {r['case_id']: r for r in read_csv(root/'case_manifest.csv')}
    expected = {r['case_id']: r for r in read_csv(root/'results_summary.csv')}
    episode_index = read_csv(root/'episode_manifest.csv')
    raw_index = {r['raw_file']: r for r in read_csv(root/'raw_manifest.csv')}
    source_episodes = {(r['case_id'],r['training_seed'],r['episode_id']): r for r in read_csv(root/'episode_results.csv')}
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
                stride = int(raw_index[relative]['association_id_stride'])
                if 'association_id_stride' in record:
                    check(int(record['association_id_stride']) == stride, 'association_encoding', raw_file=relative)
                try:
                    values = batch_prefix_metrics(record['association'], record['proposal'], record['rate_Mbps'],
                        horizons=sorted({int(e['H']) for e in entries}), capacity=10, beta_load=.8,
                        weights=tuple(float(case['weight_'+name]) for name in ['outage','ho','load','rate']),
                        n_sat=int(case['N_sat']), beams_per_satellite=stride)
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
