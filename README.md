# Portfolio Selection with Limited Holdings on the NASDAQ-100
IEOR E4004, Group 27E (Milestone 4)

Mean-absolute-deviation (MAD) portfolio with a cardinality limit, solved as a
mixed-integer linear program in Gurobi.

## How to run
```
pip install -r requirements.txt
python src/get_data.py        # optional: re-download data (saved copies are in data/)
python src/run_milestone4.py  # solve, check, evaluate; writes results/
```
All paths are relative to the repository, so the code runs on any machine.
The free pip license of Gurobi (2,000 variables / 2,000 constraints) is enough
for this model (689 variables, up to 1,192 constraints).

## Files
| File | What it does |
|---|---|
| `data/constituents.csv` | NASDAQ-100 tickers from the Nasdaq API |
| `data/prices.csv`, `data/qqq.csv` | Adjusted daily closes from Yahoo Finance, Sep 29, 2023 to Sep 30, 2026 |
| `data/dgs3mo.csv` | 3-month Treasury yield from FRED (evaluation only) |
| `src/data.py` | Builds the 94-stock universe and the estimation / test windows |
| `src/model.py` | The MILP (`solve_mad`) |
| `src/evaluate.py` | Buy-and-hold out-of-sample evaluation and metrics |
| `src/run_milestone4.py` | Runs everything and writes the results |
| `results/` | Tables (CSV) and the out-of-sample figure |

## Key settings (in `src/run_milestone4.py`)
- Baseline lambda = 1, fixed before looking at the test period
- Lambda grid for Milestone 5: 0, 0.25, 0.5, 1, 2, 5, 10
- K = 5, 10, 20, 50, and no limit; weights between 1% and 30%
- Estimation: Oct 2, 2023 to Sep 30, 2025. Test: Oct 1, 2025 to Sep 30, 2026
