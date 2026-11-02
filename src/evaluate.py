"""Out-of-sample evaluation under buy-and-hold (weights drift with prices)."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def buy_and_hold(test_rets: pd.DataFrame, w0: np.ndarray) -> pd.Series:
    """Invest w0 on the first test day and never rebalance.

    Each holding grows with its own cumulative return, so weights drift.
    Returns the daily portfolio value path starting from 1.0.
    """
    growth = (1 + test_rets).cumprod()           # value of $1 in each stock
    value = growth.values @ w0                   # portfolio value each day
    return pd.Series(value, index=test_rets.index)


def metrics(value: pd.Series, rf_daily: pd.Series | None = None) -> dict:
    v = pd.concat([pd.Series([1.0]), value.reset_index(drop=True)])
    r = v.pct_change().dropna().values
    total = value.iloc[-1] - 1
    ann_ret = (1 + total) ** (TRADING_DAYS / len(r)) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    rf = 0.0 if rf_daily is None else rf_daily.reindex(value.index).ffill().bfill().values
    excess = r - rf
    sharpe = excess.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    dd = (v / v.cummax() - 1).min()
    return {"total_return": total, "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": dd,
            "daily_mad": float(np.mean(np.abs(r - r.mean())))}
