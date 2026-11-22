"""Monthly portfolio models sharing long-only/cardinality/position constraints.

riskAdjustModel: mean/std score minus historical MAD.
robustMADModel: budgeted uncertain mean minus historical MAD.
DROModel: Wasserstein-1 worst-case mean minus historical MAD (L1 transport).
MVOModel: classical mean minus gamma times sample/Ledoit-Wolf variance.

All returns and risk measures are daily decimals. Risk-adjustment normalizes
only the reward/uncertain reward; MAD always uses actual historical returns.
"""
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def _inputs(R, K, w_min, w_max, return_mode='raw'):
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] < 2 or R.shape[1] < 1 or not np.isfinite(R).all():
        raise ValueError('R must be finite with at least two observations.')
    if not 0 <= w_min <= w_max <= 1 or w_max == 0:
        raise ValueError('Invalid weight bounds.')
    if K is not None and (int(K) != K or K < 1):
        raise ValueError('K must be a positive integer or None.')
    if return_mode not in ('raw', 'risk_adjusted'):
        raise ValueError('return_mode must be raw or risk_adjusted.')
    mu = R.mean(0)
    sigma = np.maximum(R.std(0, ddof=1), 1e-8)
    scale = np.ones(R.shape[1]) if return_mode == 'raw' else sigma
    return R, mu, sigma, mu / scale, scale


def _base(n, K, w_min, w_max, time_limit, mip_gap, verbose, threads, seed):
    m = gp.Model('portfolio')
    m.Params.OutputFlag = int(verbose)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.Threads = threads
    m.Params.Seed = seed
    x = m.addMVar(n, lb=0, ub=w_max, name='weight')
    y = m.addMVar(n, vtype=GRB.BINARY, name='selected')
    m.addConstr(x.sum() == 1, name='budget')
    m.addConstr(x >= w_min*y, name='minimum_position')
    m.addConstr(x <= w_max*y, name='maximum_position')
    if K is not None:
        m.addConstr(y.sum() <= K, name='cardinality')
    return m, x, y


def _start(x, y, score, K, w_min, w_max, warm_start):
    w = None if warm_start is None else np.asarray(warm_start, dtype=float)
    def feasible(v):
        if v is None or v.shape != score.shape or not np.isfinite(v).all(): return False
        held = v > 1e-7
        return (abs(v.sum()-1) < 1e-6 and v.min() >= -1e-8 and
                v.max() <= w_max+1e-8 and (v[held] >= w_min-1e-8).all() and
                (K is None or held.sum() <= K))
    if not feasible(w):
        # A feasible equal-weight seed avoids no-incumbent short exploratory runs.
        count = int(np.ceil(1/w_max-1e-10))
        if count <= len(score) and (K is None or count <= K) and 1/count >= w_min:
            w = np.zeros(len(score)); w[np.argsort(-score)[:count]] = 1/count
        else:
            return
    x.Start = w
    y.Start = (w > 1e-7).astype(float)


def _mad_problem(R, lam, K, w_min, w_max, return_mode, kind,
                 Gamma=0., kappa=1., epsilon=0., time_limit=60,
                 mip_gap=1e-4, verbose=False, threads=2, seed=0, warm_start=None):
    if not np.isfinite([lam, Gamma, kappa, epsilon]).all() or min(lam, Gamma, kappa, epsilon) < 0:
        raise ValueError('Penalty, uncertainty budget, kappa and epsilon must be nonnegative.')
    R, mu, sigma, reward, scale = _inputs(R, K, w_min, w_max, return_mode)
    T, n = R.shape
    m, x, y = _base(n, K, w_min, w_max, time_limit, mip_gap, verbose, threads, seed)
    _start(x, y, reward, K, w_min, w_max, warm_start)
    d = m.addMVar(T, lb=0, name='absolute_deviation')
    A = R-mu
    m.addConstr(d >= A @ x); m.addConstr(d >= -(A @ x))
    uncertainty = 0.
    delta = kappa * sigma / np.sqrt(T) / scale
    if kind == 'budgeted' and Gamma > 0 and kappa > 0:
        pi = m.addVar(lb=0, name='uncertainty_budget_price')
        q = m.addMVar(n, lb=0, name='uncertainty_slack')
        m.addConstr(pi+q >= delta*x)
        uncertainty = Gamma*pi+q.sum()
    elif kind == 'wasserstein' and epsilon > 0:
        h = m.addVar(lb=0, ub=w_max, name='largest_weight')
        m.addConstr(x <= h)
        uncertainty = epsilon*h
    # Positive scaling improves numeric magnitude without changing the optimizer.
    objective_scale = 1000.
    m.setObjective(objective_scale*(reward @ x-lam*d.sum()/T-uncertainty), GRB.MAXIMIZE)
    start = time.perf_counter(); m.optimize(); runtime = time.perf_counter()-start
    result = {'status': m.Status, 'feasible': m.SolCount > 0, 'runtime': runtime,
              'optimal': m.Status == GRB.OPTIMAL, 'n_vars': m.NumVars,
              'n_constrs': m.NumConstrs, 'return_mode': return_mode}
    if m.SolCount:
        w = x.X.copy(); port = R @ w
        mad = float(np.abs(port-port.mean()).mean())
        penalty = 0.
        if kind == 'budgeted':
            exposure = np.sort(delta*w)[::-1]
            g = min(float(Gamma), n); whole = int(np.floor(g))
            penalty = float(exposure[:whole].sum() + ((g-whole)*exposure[whole] if whole < n else 0))
        elif kind == 'wasserstein': penalty = epsilon*float(w.max())
        result.update(weights=w, selected=y.X > .5, n_hold=int((w > 1e-7).sum()),
                      objective=float(m.ObjVal)/objective_scale,
                      objective_recomputed=float(reward@w)-lam*mad-penalty,
                      objective_bound=float(m.ObjBound)/objective_scale,
                      exp_return=float(mu@w), reward=float(reward@w), mad=mad,
                      robust_penalty=penalty, gap=float(m.MIPGap))
    m.dispose()
    return result


def riskAdjustModel(R, lam, K=None, w_min=.01, w_max=.30, **kwargs):
    """Only the reward changes: (mu/sigma)'x - lam * MAD(Rx)."""
    return _mad_problem(R, lam, K, w_min, w_max, 'risk_adjusted', 'nominal', **kwargs)


def robustMADModel(R, lam, Gamma=1., kappa=1., return_mode='raw',
                   K=None, w_min=.01, w_max=.30, **kwargs):
    """Budgeted adverse mean errors: 0<=z<=1, sum(z)<=Gamma.

    delta_i=kappa*sigma_i/sqrt(T); divide both mu and delta by sigma
    in risk_adjusted mode. The standard-error approximation assumes IID days.
    """
    return _mad_problem(R, lam, K, w_min, w_max, return_mode, 'budgeted',
                        Gamma=Gamma, kappa=kappa, **kwargs)


def DROModel(R, lam, epsilon=.001, return_mode='raw',
             K=None, w_min=.01, w_max=.30, **kwargs):
    """W1 ambiguity, L1 ground cost, unrestricted support: reward-epsilon*||x||inf.

    Raw mode uses the empirical distribution of R; adjusted mode uses R/sigma
    with sigma fixed from the training window. Radius units consequently differ.
    Only expected reward is robustified; MAD stays empirical in actual returns.
    """
    return _mad_problem(R, lam, K, w_min, w_max, return_mode, 'wasserstein',
                        epsilon=epsilon, **kwargs)


def MVOModel(R, gamma=1., covariance='sample', K=None, w_min=.01, w_max=.30,
             time_limit=60, mip_gap=1e-4, verbose=False, threads=2, seed=0,
             warm_start=None):
    """Classical mu'x - gamma*x'Sigma*x, with sample or Ledoit-Wolf covariance.

    gamma is not numerically comparable to MAD lambda. No MAD term is added.
    """
    if not np.isfinite(gamma) or gamma < 0: raise ValueError('gamma must be nonnegative.')
    R, mu, sigma, reward, scale = _inputs(R, K, w_min, w_max)
    if covariance == 'sample': cov = np.atleast_2d(np.cov(R, rowvar=False, ddof=1))
    elif covariance == 'ledoit_wolf':
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(R).covariance_
    else: raise ValueError('covariance must be sample or ledoit_wolf.')
    cov = (cov+cov.T)/2
    m, x, y = _base(R.shape[1], K, w_min, w_max, time_limit, mip_gap, verbose, threads, seed)
    _start(x, y, reward, K, w_min, w_max, warm_start)
    m.setObjective(1000*(mu@x-gamma*(x@cov@x)), GRB.MAXIMIZE)
    start=time.perf_counter();m.optimize();runtime=time.perf_counter()-start
    result={'status':m.Status, 'feasible':m.SolCount>0, 'optimal':m.Status==GRB.OPTIMAL,
            'runtime':runtime, 'covariance':covariance}
    if m.SolCount:
        w=x.X.copy();port=R@w;variance=float(w@cov@w)
        result.update(weights=w,selected=y.X>.5,n_hold=int((w>1e-7).sum()),
                      objective=float(m.ObjVal)/1000,objective_bound=float(m.ObjBound)/1000,
                      objective_recomputed=float(mu@w)-gamma*variance,
                      exp_return=float(mu@w),reward=float(mu@w),variance=variance,
                      mad=float(np.abs(port-port.mean()).mean()),gap=float(m.MIPGap),
                      robust_penalty=0.)
    m.dispose()
    return result
