# Milestone 5 - Results and Sensitivity Analysis

Run all commands from the **repository root**, which contains `src/`, `data/`, and `sensiAna/`.

## 1. Base MAD and volatility-adjusted MAD

Full lambda × K grid with 252-day and 504-day estimation windows:

```bash
python sensiAna/experiments/01_lambda_k.py --lookbacks 252 504 --time-limit 2 --gap 0.01 --run-name rerun_2s
```

## 2. Budgeted Robust MAD

252-day estimation window, K=20/30, and the uncertainty-budget grid:

```bash
python sensiAna/experiments/02_robust.py --lookbacks 252 --time-limit 30 --gap 0.01 --run-name rerun_30s
```

## 3. Wasserstein DRO

Both estimation windows, K=20/30, and the Wasserstein-radius grid:

```bash
python sensiAna/experiments/03_dro.py --lookbacks 252 504 --time-limit 30 --gap 0.01 --run-name rerun_30s
```

## 4. Mean–Variance Optimization

Both estimation windows, K=10, sample and Ledoit–Wolf covariance, and the variance-penalty grid:

```bash
python sensiAna/experiments/04_mvo.py --lookbacks 252 504 --time-limit 30 --gap 0.01 --run-name rerun_30s
```

All experiments rebalance monthly. Results are saved under
`sensiAna/results/milestone5/<experiment-group>/<run-name>/`.
Repeat a command to resume; use a new `--run-name` to start a fresh run.
