"""Load prices, build the eligible universe, and split into windows."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EST_START, EST_END = "2023-10-02", "2025-09-30"    # estimation window (returns)
TEST_START, TEST_END = "2025-10-01", "2026-09-30"  # out-of-sample window (returns)

# GOOG and GOOGL are two share classes of the same company (Alphabet).
# Keeping both would let the model hold Alphabet twice, so we keep GOOGL only.
DUPLICATE_CLASSES = ["GOOG"]


def load():
    px = pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)
    qqq = pd.read_csv(DATA / "qqq.csv", index_col=0, parse_dates=True)["QQQ"]

    complete = px.columns[px.notna().all()]
    excluded = sorted(set(px.columns) - set(complete))
    universe = [t for t in complete if t not in DUPLICATE_CLASSES]

    px = px[universe]
    rets = px.pct_change().iloc[1:]          # simple daily returns
    q_rets = qqq.pct_change().iloc[1:]

    return {
        "prices": px,
        "qqq_prices": qqq,
        "est": rets.loc[EST_START:EST_END],
        "test": rets.loc[TEST_START:TEST_END],
        "qqq_test": q_rets.loc[TEST_START:TEST_END],
        "universe": universe,
        "excluded": excluded,
    }
