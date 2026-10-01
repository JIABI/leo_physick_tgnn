"""Build executable training/evaluation jobs from the 115-condition ledger."""
from pathlib import Path
import csv
import json

RULES = {'maxsinr_ttt','maxrst','greedy_analytic','cox_only'}
METHOD_ALIASES = {'mlp213':'mlp213','kan_generic':'kan_generic','full':'full',
                  'leo_madrl':'leo_madrl','no_kernel':'mlp213'}


def make_plan(data_root, output, config='code/tcom/configs/tcom.yaml'):
    root=Path(data_root)
    archive=root/'archive' if (root/'archive').exists() else root
    with (archive/'v18_event_cost_ledger_115conditions.csv').open(encoding='utf-8-sig') as f:
        ledger=list(csv.DictReader(f))
    with (archive/'01_统一主结果_115条件.csv').open(encoding='utf-8-sig') as f:
        source={r['case_id']:r for r in csv.DictReader(f)}
    train={}; evaluation=[]
    for r in ledger:
        case=r['case_id']; s=source[case]; method=r['method_id']
        weights=[float(r['weight_'+w]) for w in ['outage','ho','load','rate']]
        tau=float(r['tau_hyst']) if r['tau_hyst'] else .05
        for seed in ([0] if method in RULES else range(1,6)):
            train_id=f"{method}_N{s['train_N_sat']}_K{s['train_K']}_{r['weight_id']}_seed{seed}"
            if method not in RULES:
                train.setdefault(train_id,{'job_id':train_id,'kind':'train','method':method,'seed':seed,
                    'n_satellites':int(s['train_N_sat']),'n_users':int(s['train_K']),
                    'horizon':800,'weights':weights,'tau_hyst':.05})
            evaluation.append({'job_id':f'{case}_seed{seed}','kind':'evaluate','case_id':case,
                'method':method,'seed':seed,'n_satellites':int(r['N_sat']),'n_users':int(r['K']),
                'horizon':800,'prefixes':[int(r['H'])],'episodes':30,
                'weights':weights,'tau_hyst':tau,
                'checkpoint_job':None if method in RULES else train_id})
    # Evaluate a shared 800-step trajectory once for its requested prefixes.
    merged={}
    for job in evaluation:
        key=(job['method'],job['seed'],job['n_satellites'],job['n_users'],tuple(job['weights']),job['tau_hyst'])
        if key not in merged:
            merged[key]=dict(job,case_ids=[job['case_id']])
        else:
            merged[key]['prefixes']=sorted(set(merged[key]['prefixes']+job['prefixes']))
            merged[key]['case_ids'].append(job['case_id'])
    payload={'schema_version':1,'config':str(config),'n_archived_conditions':len(ledger),
             'training':list(train.values()),'evaluation':list(merged.values())}
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2)+'\n')
    return payload
