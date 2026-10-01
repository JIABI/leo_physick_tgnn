from pathlib import Path
import json
import numpy as np
import pytest
from cox_physick.studies import make_plan
from cox_physick.models import canonical_method,RESIDUAL_METHODS
from cox_physick.cli import repository_root
from cox_physick.fresh_results import summarize_runs


def test_all_paper_conditions_have_executable_dependencies(tmp_path):
    plan=make_plan(repository_root()/'data/tcom',tmp_path/'plan.json')
    assert plan['n_archived_conditions']==115
    assert len(plan['training'])==190
    assert len(plan['evaluation'])==315
    training={r['job_id']:r for r in plan['training']}
    ids=set()
    for job in plan['evaluation']:
        ids.update(job['case_ids'])
        method=canonical_method(job['method'])
        if job['checkpoint_job']:
            fitted=training[job['checkpoint_job']]
            assert fitted['n_users']==100
            assert fitted['n_satellites']==job['n_satellites']
            assert fitted['weights']==job['weights']
            assert method in RESIDUAL_METHODS|{'madrl'}
            assert fitted['tau_hyst']==.05
        assert max(job['prefixes'])<=job['horizon']==800
    assert len(ids)==115


def _run(folder,method,training_seed,values):
    import csv
    folder.mkdir()
    env=dict(n_satellites=2000,n_users=100,hysteresis=.05)
    meta=dict(method=method,checkpoint='selected.pt' if training_seed is not None else None,
              training_seed=training_seed,config=dict(environment=env,weights=[5,.5,.5,1]))
    (folder/'run.json').write_text(json.dumps(meta))
    with (folder/'episode_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,['method','environment_seed','episode_id','H','raw_file','J_user'])
        w.writeheader()
        for i,v in enumerate(values):
            w.writerow(dict(method=method,environment_seed=20001,episode_id=i,H=200,raw_file='x.npz',J_user=v))


def test_fresh_seed_first_and_fixed_greedy_contrast(tmp_path):
    runs=tmp_path/'runs';runs.mkdir()
    _run(runs/'greedy','greedy_analytic',None,[-1,-.8])
    for seed,vals in enumerate([[-1,-.9],[-.8,-1.]],1):_run(runs/f'full{seed}','full',seed,vals)
    summary=summarize_runs(runs,tmp_path/'summary')
    assert summary['conditions']==2
    assert summary['cost_contrasts']==1
    import csv
    rows=list(csv.DictReader((tmp_path/'summary/condition_summary.csv').open()))
    full=next(r for r in rows if r['method_id']=='full')
    assert int(full['n_summary_units'])==2
    assert float(full['J_user_mean'])==pytest.approx(-.925)
    assert float(full['J_user_sample_sd'])==pytest.approx(np.std([-.95,-.9],ddof=1))


def test_duplicate_episodes_are_not_independent_repetitions(tmp_path):
    runs=tmp_path/'runs';runs.mkdir()
    _run(runs/'a','full',1,[-1,-.9]);_run(runs/'duplicate','full',1,[-1,-.9])
    with pytest.raises(ValueError,match='Duplicate'):
        summarize_runs(runs,tmp_path/'summary')
