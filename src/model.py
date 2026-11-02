"""Mean-absolute-deviation (MAD) portfolio with a cardinality limit (MILP).

    max   sum_i mu_i x_i - (lam / T) sum_t d_t
    s.t.  sum_i x_i = 1
          w_min y_i <= x_i <= w_max y_i          for all i
          sum_i y_i <= K
          d_t >=  sum_i a_it x_i                  for all t
          d_t >= -sum_i a_it x_i                  for all t
          x_i >= 0, d_t >= 0, y_i binary

a_it = r_it - mu_i are centered daily returns. K=None drops the cardinality
constraint (the unconstrained comparison, other restrictions unchanged).
"""
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def solve_mad(R, lam, K=None, w_min=0.01, w_max=0.30,
              time_limit=60, mip_gap=1e-4, verbose=False):
    """R: (T x n) numpy array of daily simple returns (estimation window).

    Returns dict with weights, objective parts, solver status, runtime, gap.
    """
    T, n = R.shape
    mu = R.mean(axis=0)
    A = R - mu  # centered deviations a_it

    m = gp.Model("mad_cardinality")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap

    x = m.addMVar(n, lb=0.0, ub=1.0, name="x")
    y = m.addMVar(n, vtype=GRB.BINARY, name="y")
    d = m.addMVar(T, lb=0.0, name="d")

    m.addConstr(x.sum() == 1, name="budget")                 # (1)
    m.addConstr(x <= w_max * y, name="cap_link")             # (2)
    m.addConstr(x >= w_min * y, name="min_pos")              # (3)
    if K is not None:
        m.addConstr(y.sum() <= K, name="cardinality")        # (4)
    m.addConstr(d >= A @ x, name="dev_pos")                  # (5)
    m.addConstr(d >= -(A @ x), name="dev_neg")               # (6)

    m.setObjective(mu @ x - (lam / T) * d.sum(), GRB.MAXIMIZE)

    t0 = time.time()
    m.optimize()
    runtime = time.time() - t0

    if m.SolCount == 0:
        return {"status": m.Status, "feasible": False, "runtime": runtime}

    w = x.X.copy()
    w[w < 1e-9] = 0.0
    port = R @ w
    mad = np.mean(np.abs(port - port.mean()))
    return {
        "status": m.Status,
        "feasible": True,
        "optimal": m.Status == GRB.OPTIMAL,
        "weights": w,
        "selected": y.X > 0.5,
        "n_hold": int((y.X > 0.5).sum()),
        "objective": m.ObjVal,
        "exp_return": float(mu @ w),
        "mad": float(mad),
        "gap": float(m.MIPGap) if m.IsMIP else 0.0,
        "runtime": runtime,
        "n_vars": m.NumVars,
        "n_constrs": m.NumConstrs,
    }
