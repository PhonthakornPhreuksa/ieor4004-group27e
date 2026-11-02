"""Milestone 4: solve the baseline model for each K, run sanity checks,
and evaluate out of sample (buy-and-hold). Writes outputs to ../results.

Run:  python src/run_milestone4.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import DATA, load
from evaluate import buy_and_hold, metrics
from model import solve_mad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Fixed before looking at the test period (CA feedback, Milestone 3).
BASELINE_LAMBDA = 1.0
LAMBDA_GRID = [0, 0.25, 0.5, 1, 2, 5, 10]   # for Milestone 5 sensitivity
K_VALUES = [5, 10, 20, 50, None]            # None = no cardinality limit
W_MIN, W_MAX = 0.01, 0.30
TOL = 1e-6
TIME_LIMIT = 300  # seconds per solve


def check_solution(res, R, K, lam):
    """Sanity checks on one solution. Returns list of failed checks."""
    w, sel = res["weights"], res["selected"]
    fails = []
    if abs(w.sum() - 1) > TOL:
        fails.append("weights do not sum to 1")
    if (w < -TOL).any():
        fails.append("negative weight")
    held = w > TOL
    if (w[held] < W_MIN - TOL).any() or (w[held] > W_MAX + TOL).any():
        fails.append("weight outside [1%, 30%]")
    if (held & ~sel).any():
        fails.append("weight on unselected stock")
    if K is not None and sel.sum() > K:
        fails.append("too many holdings")
    # objective recomputed independently from the weights
    mu = R.mean(0)
    port = R @ w
    obj = mu @ w - lam * np.mean(np.abs(port - port.mean()))
    if abs(obj - res["objective"]) > 1e-7:
        fails.append(f"objective mismatch {obj:.3e} vs {res['objective']:.3e}")
    return fails


def main():
    D = load()
    R = D["est"].values
    tick = np.array(D["universe"])
    rf = pd.read_csv(DATA / "dgs3mo.csv", index_col=0, parse_dates=True).iloc[:, 0]
    rf_daily = (rf / 100) / 252

    # ---- data summary (estimation window) ----
    mu = R.mean(0)
    mad_i = np.abs(R - mu).mean(0)
    summary = pd.DataFrame({"mu_daily": mu, "mad_daily": mad_i}, index=tick)
    summary.to_csv(OUT / "parameter_summary.csv")
    print(f"Universe n={len(tick)}, estimation T={R.shape[0]}, "
          f"test T={len(D['test'])}, excluded={D['excluded']}")
    print(summary.describe().loc[["min", "50%", "max"]])

    # ---- baseline solves ----
    rows, weights, values = [], {}, {}
    for K in K_VALUES:
        res = solve_mad(R, BASELINE_LAMBDA, K=K, w_min=W_MIN, w_max=W_MAX,
                        time_limit=TIME_LIMIT)
        label = "No limit" if K is None else f"K={K}"
        fails = check_solution(res, R, K, BASELINE_LAMBDA)
        v = buy_and_hold(D["test"], res["weights"])
        oos = metrics(v, rf_daily)
        values[label] = v
        weights[label] = pd.Series(res["weights"], index=tick)
        rows.append({"portfolio": label, "K": K, "n_hold": res["n_hold"],
                     "objective": res["objective"],
                     "exp_return_ann": res["exp_return"] * 252,
                     "mad_daily_in": res["mad"],
                     "runtime_s": res["runtime"], "gap": res["gap"],
                     "optimal": res["optimal"], "checks_failed": "; ".join(fails),
                     **{f"oos_{k}": val for k, val in oos.items()}})

    # ---- benchmarks (same buy-and-hold rule) ----
    n = len(tick)
    for label, v in [("Equal weight", buy_and_hold(D["test"], np.full(n, 1 / n))),
                     ("QQQ", (1 + D["qqq_test"]).cumprod())]:
        values[label] = v
        rows.append({"portfolio": label,
                     **{f"oos_{k}": val for k, val in metrics(v, rf_daily).items()}})

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT / "baseline_results.csv", index=False)
    pd.DataFrame(weights).to_csv(OUT / "baseline_weights.csv")
    pd.DataFrame(values).to_csv(OUT / "oos_values.csv")

    # ---- sanity check: MILP vs. a "keep the K largest" shortcut ----
    # Shortcut: take the K largest weights from the no-limit solution, then
    # re-optimize weights on only those K stocks. The MILP searches over all
    # subsets, so its objective must be at least as good.
    free = weights["No limit"].values
    naive_rows = []
    for K in [5, 10, 20]:
        top = np.argsort(-free)[:K]
        sub = solve_mad(R[:, top], BASELINE_LAMBDA, K=None, w_min=W_MIN, w_max=W_MAX)
        milp = res_df.loc[res_df.K == K, "objective"].item()
        naive_rows.append({"K": K, "milp_objective": milp,
                           "shortcut_objective": sub["objective"],
                           "milp_better_by_pct": 100 * (milp - sub["objective"]) / abs(sub["objective"])})
    pd.DataFrame(naive_rows).to_csv(OUT / "check_naive_vs_milp.csv", index=False)

    # ---- how many stocks does the model hold with no limit, by lambda ----
    lam_rows = []
    for lam in LAMBDA_GRID:
        r = solve_mad(R, lam, K=None, w_min=W_MIN, w_max=W_MAX)
        lam_rows.append({"lambda": lam, "n_hold_no_limit": r["n_hold"],
                         "exp_return_ann": r["exp_return"] * 252,
                         "mad_daily": r["mad"]})
    pd.DataFrame(lam_rows).to_csv(OUT / "holdings_by_lambda.csv", index=False)

    # ---- figure: out-of-sample value paths ----
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for label, v in values.items():
        if label == "K=50":   # identical to "No limit" (limit does not bind)
            continue
        style = "--" if label in ("QQQ", "Equal weight") else "-"
        ax.plot(v.index, v.values, style, lw=1.4, label=label)
    ax.set_ylabel("Value of $1 (buy-and-hold)")
    ax.set_title(f"Out-of-sample, Oct 2025 to Sep 2026 ($\\lambda$ = {BASELINE_LAMBDA:g})")
    ax.legend(fontsize=8, ncol=4, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "oos_value.png", dpi=200)

    pd.set_option("display.width", 200)
    print(res_df.round(4).to_string())
    print(pd.DataFrame(naive_rows))
    print(pd.DataFrame(lam_rows))


if __name__ == "__main__":
    main()
