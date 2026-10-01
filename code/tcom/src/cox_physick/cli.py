"""One command line for TCOM archives, physical rollouts, and training."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def json_default(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=json_default,allow_nan=True)+'\n')


def repository_root():
    for p in Path(__file__).resolve().parents:
        if (p/'code/tcom/pyproject.toml').exists(): return p
    return Path.cwd()


def _config(args):
    from .config import Config
    cfg=Config.from_yaml(args.config)
    for arg,key in [('satellites','n_satellites'),('users','n_users'),('horizon','horizon'),('hysteresis','hysteresis')]:
        if getattr(args,arg,None) is not None: cfg.environment[key]=getattr(args,arg)
    if getattr(args,'weights',None): cfg.weights=tuple(args.weights)
    cfg.__post_init__()
    return cfg


def evaluate(cfg, method, output, *, checkpoint=None, seed=20001, episodes=30,
             start_episode=0, prefixes=None, device='cpu'):
    """Execute fresh trajectories and write raw arrays plus episode metrics."""
    from .training import evaluate_episode, load_controller
    from .controllers import make_controller
    from .metrics import batch_prefix_metrics
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    controller=load_controller(checkpoint,cfg,device) if checkpoint else make_controller(method,cfg,device=device)
    from .models import canonical_method
    if checkpoint and canonical_method(method) != canonical_method(controller.method):
        raise ValueError('Requested method does not match the loaded model')
    cfg=controller.cfg
    rows=[]
    for episode in range(start_episode,start_episode+episodes):
        from .environment import Environment
        env=Environment(cfg,seed,episode)
        raw=evaluate_episode(controller,cfg,seed,episode,environment=env,record_diagnostics=True)
        env.export_exogenous(output/f'episode_{episode:04d}_exogenous.npz')
        arrays={k:v for k,v in raw.items() if isinstance(v,np.ndarray)}
        raw_path=output/f'episode_{episode:04d}.npz'
        np.savez_compressed(raw_path,**arrays)
        a=arrays['association'];q=arrays['proposal'];rate=arrays['rate_mbps']
        metrics=batch_prefix_metrics(a[:,None,:],q[:,None,:],rate[:,None,:],horizons=prefixes,
            capacity=cfg.environment['capacity'],beta_load=cfg.environment['peak_smoothing'],
            dt_s=cfg.environment['dt'],weights=cfg.weights,rate_reference=cfg.environment['r_ref_mbps'],
            n_sat=cfg.environment['n_satellites'],beams_per_satellite=cfg.environment['n_beams'],
            c_interbeam=cfg.environment['c_interbeam'],c_intersatellite=cfg.environment['c_intersatellite'])
        for h,values in metrics.items():
            rows.append(dict(method=method,environment_seed=seed,episode_id=episode,H=h,
                             raw_file=raw_path.name,**{k:float(v[0]) for k,v in values.items()}))
        print(f'episode {episode+1-start_episode}/{episodes}: cost={raw["mean_cost"]:.6f}',flush=True)
    with (output/'episode_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(output/'run.json',dict(method=method,config=cfg.to_dict(),environment_seed=seed,
         episode_ids=list(range(start_episode,start_episode+episodes)),training_seed=getattr(controller,'training_seed',None),checkpoint=str(checkpoint) if checkpoint else None,
         association_stride=cfg.environment['n_beams'],result_kind='fresh_execution'))
    return rows


def run_plan(plan_path, output, *, job_ids=None, stage='all',device='cpu'):
    from .config import Config
    from .training import train_residual, train_dqn
    from .models import canonical_method
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config_path=Path(plan['config'])
    if not config_path.is_absolute(): config_path=repository_root()/config_path
    selected=set(job_ids or []);completed=[]
    index_path=output/'checkpoint_index.json'
    index=json.loads(index_path.read_text()) if index_path.exists() else {}
    for job in plan['training']+plan['evaluation']:
        if stage!='all' and job['kind']!=stage: continue
        if selected and job['job_id'] not in selected: continue
        cfg=Config.from_yaml(config_path)
        cfg.environment.update({k:job[k] for k in ['n_satellites','n_users','horizon']})
        cfg.environment['hysteresis']=job['tau_hyst'];cfg.weights=tuple(job['weights']);cfg.__post_init__()
        dest=output/job['kind']/job['job_id']
        if job['kind']=='train':
            if canonical_method(job['method'])=='madrl':
                checkpoint=train_dqn(cfg,job['seed'],dest,device=device)
            else:
                checkpoint=train_residual(cfg,job['method'],job['seed'],dest,device=device)
            index[job['job_id']]=str(Path(checkpoint).resolve());dump(index_path,index)
        else:
            key=job['checkpoint_job']
            if key and key not in index: raise FileNotFoundError(f'Train {key} first; no checkpoint is registered in {index_path}')
            evaluate(cfg,job['method'],dest,checkpoint=index.get(key),episodes=job['episodes'],
                     prefixes=job['prefixes'],device=device)
        completed.append(job['job_id']);dump(output/'completed_jobs.json',completed)
    return {'completed_jobs':completed}


def main(argv=None):
    root=repository_root()
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('inventory',help='Implemented models and exact module parameter counts')
    for name in ['audit','calibrate','plots','reproduce','plan']:
        s=sub.add_parser(name)
        s.add_argument('--data-root',type=Path,default=root/'data/tcom')
        s.add_argument('--output',type=Path,default=root/'results/tcom'/name)
        if name in ['audit','reproduce']: s.add_argument('--full',action='store_true',help='Recompute all raw execution prefixes')
    for name in ['train','evaluate']:
        s=sub.add_parser(name)
        s.add_argument('--config',type=Path,default=root/'code/tcom/configs/tcom.yaml')
        s.add_argument('--method',required=True)
        s.add_argument('--output',type=Path,required=True)
        s.add_argument('--seed',type=int,default=1 if name=='train' else 20001)
        s.add_argument('--device',default='cpu')
        s.add_argument('--satellites',type=int);s.add_argument('--users',type=int);s.add_argument('--horizon',type=int)
        s.add_argument('--hysteresis',type=float);s.add_argument('--weights',type=float,nargs=4)
        if name=='train':
            s.add_argument('--overrides',type=Path,help='JSON training overrides, recorded with outputs')
            s.add_argument('--resume',type=Path)
        else:
            s.add_argument('--checkpoint',type=Path)
            s.add_argument('--episodes',type=int,default=30)
            s.add_argument('--start-episode',type=int,default=0)
            s.add_argument('--prefixes',nargs='+',type=int)
    s=sub.add_parser('summarize-runs',help='Aggregate fresh outputs by episode and training seed')
    s.add_argument('--runs',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('run-plan',help='Execute listed training/evaluation jobs')
    s.add_argument('--plan',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    s.add_argument('--job',action='append');s.add_argument('--stage',choices=['all','train','evaluate'],default='all');s.add_argument('--device',default='cpu')
    args=p.parse_args(argv)
    if args.command=='inventory':
        from .models import parameter_counts, RESIDUAL_METHODS
        result={'version':'22.0.0','residual_models':sorted(RESIDUAL_METHODS),
                'rule_controllers':['maxsinr_ttt','maxrst','greedy_analytic','cox_only'],
                'drl':'adapted parameter-sharing hysteretic DQN','parameter_counts':parameter_counts()}
    elif args.command in ['audit','reproduce']:
        from .results import audit_results
        result=audit_results(args.data_root,args.output,full=args.full)
        if args.command=='reproduce':
            from .calibration import recompute_calibration
            from .plotting import plot_results
            recompute_calibration(args.data_root,args.output/'calibration')
            plot_results(args.data_root,args.output/'figures')
    elif args.command=='calibrate':
        from .calibration import recompute_calibration
        result=recompute_calibration(args.data_root,args.output)
    elif args.command=='plots':
        from .plotting import plot_results
        result=plot_results(args.data_root,args.output)
    elif args.command=='plan':
        from .studies import make_plan
        dest=args.output if args.output.suffix=='.json' else args.output/'experiment_plan.json'
        plan=make_plan(args.data_root,dest)
        result={'plan':dest,'training_jobs':len(plan['training']),'evaluation_jobs':len(plan['evaluation'])}
    elif args.command=='train':
        from .training import train_residual,train_dqn
        from .models import canonical_method
        cfg=_config(args);options=json.loads(args.overrides.read_text()) if args.overrides else {}
        if canonical_method(args.method)=='madrl':
            checkpoint=train_dqn(cfg,args.seed,args.output,device=args.device,overrides=options,resume=args.resume)
        else:
            checkpoint=train_residual(cfg,args.method,args.seed,args.output,device=args.device,overrides=options,resume=args.resume)
        result={'checkpoint':checkpoint}
    elif args.command=='summarize-runs':
        from .fresh_results import summarize_runs
        result=summarize_runs(args.runs,args.output)
    elif args.command=='evaluate':
        result={'rows':len(evaluate(_config(args),args.method,args.output,checkpoint=args.checkpoint,
                     seed=args.seed,episodes=args.episodes,start_episode=args.start_episode,
                     prefixes=args.prefixes,device=args.device)),'output':args.output}
    else:
        result=run_plan(args.plan,args.output,job_ids=args.job,stage=args.stage,device=args.device)
    print(json.dumps(result,indent=2,default=json_default,allow_nan=True))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
