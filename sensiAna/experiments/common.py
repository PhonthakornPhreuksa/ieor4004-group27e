"""Shared monthly rolling engine; preserves drifting weights between trades."""
from pathlib import Path
import sys,json,time,argparse,inspect,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'src'))
from data import load, DATA
from evaluate import metrics
# Load base MAD explicitly from the existing repository, avoiding module-name collisions.
_spec = importlib.util.spec_from_file_location("milestone4_model", ROOT.parent/'src'/'model.py')
_base_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base_model)
solve_mad = _base_model.solve_mad
sys.path.insert(0, str(ROOT))
from updateModel import riskAdjustModel,robustMADModel,DROModel,MVOModel
MODELS={f.__name__:f for f in [solve_mad,riskAdjustModel,robustMADModel,DROModel,MVOModel]}
LAMBDAS=[0,.25,.5,1,2,5,10]
KS=[None,50,20,10,5]


def _solve(model_name, R, **kwargs):
    """Adapt the existing repository's base-model interface for monthly logging."""
    solver = MODELS[model_name]
    parameters = inspect.signature(solver).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    supported = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in parameters}
    result = dict(solver(R, **supported))
    if model_name != 'solve_mad' or not result['feasible']:
        return result
    w = np.asarray(result['weights'])
    mu = R.mean(axis=0)
    mad = float(np.abs((R-mu) @ w).mean())
    objective = float(mu @ w)-kwargs['lam']*mad
    result.setdefault('objective_recomputed', objective)
    result.setdefault('reward', float(mu @ w))
    result.setdefault('robust_penalty', 0.)
    result.setdefault('return_mode', 'raw')
    result['n_hold'] = int((w > 1e-7).sum())
    # Original API exposes the relative MIP gap rather than ObjBound. For a
    # maximization problem, UB = incumbent + gap * abs(incumbent).
    if 'objective_bound' not in result:
        gap = float(result.get('gap', float('inf')))
        obj = float(result['objective'])
        result['objective_bound'] = (obj if gap == 0 else
            obj + gap*abs(obj) if np.isfinite(gap) and obj != 0 else float('inf'))
    return result


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--lookbacks',nargs='+',type=int,default=[252,504])
    p.add_argument('--time-limit',type=float,default=2.)
    p.add_argument('--gap',type=float,default=.01)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--cost-bps',type=float,default=0.)
    p.add_argument('--run-name',default='initial')
    return p


def cases(group,lookbacks):
    out=[]
    for lookback in lookbacks:
        if group=='01_lambda_k':
            for model in ['solve_mad','riskAdjustModel']:
                for lam in LAMBDAS:
                    for K in KS: out.append(dict(model=model,lookback=lookback,lam=lam,K=K))
        elif group=='02_robust':
            for mode in ['raw','risk_adjusted']:
                for K in [20,30]:
                    for Gamma in [0,1,3,5,10,K]:
                        out.append(dict(model='robustMADModel',lookback=lookback,lam=1.,K=K,return_mode=mode,Gamma=Gamma,kappa=1.))
        elif group=='03_dro':
            for mode,grid in [('raw',[0,.0005,.001,.002,.005]),('risk_adjusted',[0,.025,.05,.1,.25])]:
                for K in [20,30]:
                    for epsilon in grid: out.append(dict(model='DROModel',lookback=lookback,lam=1.,K=K,return_mode=mode,epsilon=epsilon))
        elif group=='04_mvo':
            for covariance in ['sample','ledoit_wolf']:
                for gamma in [0,1,5,10,25,50,100]:
                    out.append(dict(model='MVOModel',lookback=lookback,K=10,gamma=gamma,covariance=covariance))
    return out


def run(group,args):
    out=ROOT/'results'/'milestone5'/group/args.run_name
    out.mkdir(parents=True,exist_ok=True)
    D=load(); all_returns=pd.concat([D['est'],D['test']]).sort_index()
    test=D['test']; tick=list(test.columns)
    if not test.index.equals(D['qqq_test'].index): raise ValueError('Benchmark dates differ')
    rf=pd.read_csv(DATA/'dgs3mo.csv',index_col=0,parse_dates=True).iloc[:,0]/100/252
    grid=cases(group,args.lookbacks)
    config=dict(group=group,**vars(args),grid=grid,w_min=.01,w_max=.30,
                test_start=str(test.index[0].date()),test_end=str(test.index[-1].date()),
                rebalance='first return date of each calendar month; execute at preceding close',
                history_policy='min(available, requested): first 504-day window has 501 observations',
                initial_formation_cost='excluded',seed=0)
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Run name already exists with different settings; use --run-name.')
    config_path.write_text(json.dumps(config,indent=2))
    qvalue=(1+D['qqq_test']).cumprod()
    pd.DataFrame([dict(strategy='QQQ',**metrics(qvalue,rf))]).to_csv(out/'benchmark.csv',index=False)
    base_fits={}; summary=[]
    months=test.index.to_period('M')
    month_starts={date for i,date in enumerate(test.index) if i==0 or months[i]!=months[i-1]}
    for number,case in enumerate(grid,1):
        case_id='__'.join(f'{k}-{v}' for k,v in case.items())
        dest=out/case_id;dest.mkdir(exist_ok=True)
        summary_path=dest/'summary.json'
        if summary_path.exists():
            row=json.loads(summary_path.read_text());summary.append(row)
            if group=='01_lambda_k' and case['K'] is None and row['completed']:
                logs=pd.read_csv(dest/'rebalances.csv'); weights=pd.read_csv(dest/'weights.csv',index_col=0,parse_dates=True)
                for rec in logs.to_dict('records'):
                    date=pd.Timestamp(rec['date']); rec['weights']=weights.loc[date].to_numpy()
                    base_fits[(case['model'],case['lookback'],case['lam'],date)]=rec
            print(f'[{number}/{len(grid)}] cached {case_id}',flush=True)
            continue
        start=time.perf_counter();w=None;records=[];weight_records=[];daily=[];failure=None
        for date in test.index:
            turnover=0.;cost=0.
            if date in month_starts:
                hist=all_returns.loc[all_returns.index<date].tail(case['lookback'])
                params={k:v for k,v in case.items() if k not in ['model','lookback']}
                key=(case['model'],case['lookback'],case.get('lam'),date)
                relaxed=base_fits.get(key) if group=='01_lambda_k' and case['K'] is not None else None
                reuse=relaxed is not None and int(relaxed['n_hold'])<=case['K']
                if reuse:
                    result=dict(relaxed);result['runtime']=0.;result['feasible']=True
                else:
                    result=_solve(case['model'],hist.to_numpy(),**params,w_min=.01,w_max=.30,
                        time_limit=args.time_limit,mip_gap=args.gap,threads=args.threads,seed=0,warm_start=w)
                if not result['feasible']:
                    failure=f'{date.date()}: solver status {result["status"]}; no incumbent'
                    break
                new_w=np.asarray(result['weights']).copy()
                held=new_w>1e-7
                if abs(new_w.sum()-1)>1e-5 or new_w.min()<-1e-6 or new_w.max()>.30001 or (new_w[held]<.00999).any() or (case['K'] is not None and held.sum()>case['K']):
                    raise RuntimeError(f'Invalid portfolio at {date}: {case_id}')
                if abs(result['objective']-result['objective_recomputed'])>1e-5:
                    raise RuntimeError('Objective recomputation mismatch')
                turnover=0. if w is None else float(abs(new_w-w).sum())
                w=new_w
                record={k:v for k,v in result.items() if k not in ['weights','selected','feasible']}
                record.update(date=str(date.date()),actual_lookback=len(hist),training_start=str(hist.index[0].date()),training_end=str(hist.index[-1].date()),
                              turnover=turnover,reused_no_limit=reuse)
                records.append(record)
                if group == '02_robust':
                    print(f'  {case_id} {date.date()} gap={result["gap"]:.4g} time={result["runtime"]:.1f}s', flush=True)
                weight_records.append(dict(date=str(date.date()),**dict(zip(tick,w))))
                if group=='01_lambda_k' and case['K'] is None:
                    base_fits[key]={**record,'weights':w.copy()}
            r=test.loc[date].to_numpy();gross=float(r@w)
            cost=turnover*args.cost_bps*1e-4
            net=(1-cost)*(1+gross)-1
            daily.append(dict(date=str(date.date()),net_return=net,gross_return=gross,turnover=turnover,cost_fraction=cost))
            w=w*(1+r)/(1+gross)
        logs=pd.DataFrame(records);logs.to_csv(dest/'rebalances.csv',index=False)
        pd.DataFrame(weight_records).to_csv(dest/'weights.csv',index=False)
        daily=pd.DataFrame(daily)
        if len(daily):
            daily['value']=(1+daily.net_return).cumprod();daily.to_csv(dest/'daily.csv',index=False)
        row=dict(case_id=case_id,**case,completed=failure is None,failure=failure,
                 elapsed_s=time.perf_counter()-start)
        if failure is None:
            value=pd.Series(daily.value.to_numpy(),index=test.index)
            row.update(**metrics(value,rf),n_solves=len(logs),nonoptimal=int((~logs.optimal.astype(bool)).sum()),
                       max_gap=float(logs.gap.max()),mean_gap=float(logs.gap.mean()),mean_hold=float(logs.n_hold.mean()),
                       total_turnover=float(daily.turnover.sum()),solve_s=float(logs.runtime.sum()),
                       reused_no_limit=int(logs.reused_no_limit.sum()),mean_objective=float(logs.objective.mean()),
                       mean_robust_penalty=float(logs.robust_penalty.mean()),mean_reward=float(logs.reward.mean()),mean_exp_return=float(logs.exp_return.mean()),mean_mad_in=float(logs['mad'].mean()),
                       min_actual_lookback=int(logs.actual_lookback.min()))
        summary_path.write_text(json.dumps(row,indent=2))
        summary.append(row);pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
        print(f'[{number}/{len(grid)}] {case_id}: '+(f'return={row["total_return"]:.2%}, nonoptimal={row["nonoptimal"]}, maxgap={row["max_gap"]:.3g}, {row["elapsed_s"]:.1f}s' if failure is None else failure),flush=True)
    frame=pd.DataFrame(summary);frame.to_csv(out/'summary.csv',index=False)
    if group=='01_lambda_k':
        comparisons=[]
        for case in grid:
            if case['K'] is None: continue
            cid='__'.join(f'{k}-{v}' for k,v in case.items())
            free=dict(case,K=None);fid='__'.join(f'{k}-{v}' for k,v in free.items())
            a=pd.read_csv(out/fid/'rebalances.csv'); b=pd.read_csv(out/cid/'rebalances.csv')
            for _,r in a.merge(b,on='date',suffixes=('_free','_K')).iterrows():
                diff=r.objective_free-r.objective_K
                comparisons.append(dict(**case,date=r.date,objective_free=r.objective_free,objective_K=r.objective_K,
                    objective_cost=diff,percent_cost=100*diff/abs(r.objective_free) if abs(r.objective_free)>1e-8 else np.nan,
                    reward_difference=r.reward_free-r.reward_K,raw_return_difference=r.exp_return_free-r.exp_return_K,
                    mad_difference=r.mad_free-r.mad_K,gap_free=r.gap_free,gap_K=r.gap_K,
                    cost_lower=r.objective_free-r.objective_bound_K,cost_upper=r.objective_bound_free-r.objective_K))
        pd.DataFrame(comparisons).to_csv(out/'cardinality_cost.csv',index=False)
    print('Saved results:', out, flush=True)
    return frame
