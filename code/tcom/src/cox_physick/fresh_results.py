"""Aggregate new physical runs with the manuscript's independent-unit convention."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import numpy as np
from scipy.stats import t as student_t
from .metrics import summarize

EXCLUDE={'method','environment_seed','episode_id','H','raw_file'}


def _write(path, rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def summarize_runs(runs, output):
    runs,output=Path(runs),Path(output);output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for manifest in sorted(runs.rglob('run.json')):
        meta=json.loads(manifest.read_text());cfg=meta['config'];env=cfg['environment']
        with (manifest.parent/'episode_metrics.csv').open() as f: rows=list(csv.DictReader(f))
        for r in rows:
            key=(meta['method'],env['n_satellites'],env['n_users'],int(r['H']),
                 tuple(cfg['weights']),env['hysteresis'])
            groups[key].append((r,meta,manifest))
    if not groups: raise FileNotFoundError(f'No TCOM run.json and episode_metrics.csv under {runs}')
    wide=[];long=[];cost_units={};ensemble_test_ids={}
    for key,entries in groups.items():
        method,N,K,H,weights,tau=key
        learned=any(meta['checkpoint'] is not None for _,meta,_ in entries)
        if learned and any(meta.get('training_seed') is None for _,meta,_ in entries):
            raise ValueError(f'{key}: learned results need recorded training_seed')
        if any((meta['checkpoint'] is not None)!=learned for _,meta,_ in entries):
            raise ValueError('Mixed learned and deterministic identities')
        seen=set();traces=defaultdict(set)
        for r,meta,path in entries:
            uid=(meta.get('training_seed') if learned else None,int(r['environment_seed']),int(r['episode_id']))
            if uid in seen: raise ValueError(f'Duplicate episode/prefix: {path} {uid} H={H}')
            seen.add(uid);traces[uid[0]].add(uid[1:])
        if learned and len({frozenset(v) for v in traces.values()}) != 1:
            raise ValueError('Training seeds must use the same exogenous test episodes for this aggregation')
        base=dict(method_id=method,N_sat=N,K=K,H=H,weight_outage=weights[0],weight_ho=weights[1],
                  weight_load=weights[2],weight_rate=weights[3],tau_hyst=tau,
                  aggregation_unit='training_seed_mean' if learned else 'episode',
                  n_episodes=len(entries),n_summary_units=len(traces) if learned else len(entries))
        row=dict(base)
        for metric in entries[0][0]:
            if metric in EXCLUDE: continue
            values=[float(r[metric]) for r,_,_ in entries]
            ids=[meta.get('training_seed') for _,meta,_ in entries] if learned else None
            stats=summarize(values,ids)
            row[metric+'_mean']=stats['mean'];row[metric+'_sample_sd']=stats['sample_sd']
            long.append(dict(base,metric=metric,**stats))
        wide.append(row)
        unit_values=defaultdict(list)
        for r,meta,_ in entries:
            unit_values[meta.get('training_seed') if learned else None].append(float(r['J_user']))
        cost_units[key]={seed:float(np.mean(vals)) for seed,vals in unit_values.items()}
        ensemble_test_ids[key]=next(iter(traces.values()))
    contrasts=[]
    for key,units in cost_units.items():
        method,N,K,H,w,tau=key
        if method not in ('full','cox_physick','cox-physick'): continue
        greedy_key=('greedy_analytic',N,K,H,w,tau)
        if greedy_key not in cost_units: continue
        if ensemble_test_ids[key] != ensemble_test_ids[greedy_key]:
            raise ValueError('A cost contrast needs the same exogenous test episode set')
        delta=np.array(list(units.values()))-cost_units[greedy_key][None]
        n=len(delta);mean=float(delta.mean())
        half=float(student_t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
        contrasts.append(dict(method=method,reference='greedy_analytic',N_sat=N,K=K,H=H,
                              tau_hyst=tau,weights=json.dumps(w),mean_delta_J=mean,
                              t_interval_lower=mean-half,t_interval_upper=mean+half,n_training_seeds=n))
    _write(output/'condition_summary.csv',wide);_write(output/'metric_summary_long.csv',long)
    _write(output/'cost_contrasts_vs_greedy.csv',contrasts)
    report={'conditions':len(groups),'episode_prefixes':sum(map(len,groups.values())),
            'aggregation':'valid episode means followed by sample SD across training-seed means; rules use episode SD',
            'cost_contrasts':len(contrasts)}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
