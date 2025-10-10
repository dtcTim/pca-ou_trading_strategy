# OU Factor-Hedged FX Backtest (PCA)

**What it is (overview)**  
FX mean-reversion (OU) strategy where prices are regressed on **lagged PCA factors** and signals come from the **residual z-score**.  
A **delta_PC hedge** neutralizes common FX risk so PnL reflects mean reversion, not broad dollar/carry moves.  
Backtest is **trade level with costs** (commission + spread) and **Bayesian + fractional Kelly sizing**; robustness is checked via a **Monte-Carlo bootstrap** of trade returns.

---

## How it works (pipeline)

1. **Data** → Parquet OHLC per timeframe/ticker (`data_*` folders).  
2. **Factors** → `hypothesis_pca/pc3.py` builds principal components: `pca(timeframe) → DF[PC1..]`.  
3. **Signals** → `Backtest.py` regresses price on **lagged PCs**, computes **OU residual z-score**, goes long/short when `|z| > threshold`.  
4. **Hedge** → Ridge replica of ΔPC applied to the instrument’s return (factor-neutral PnL).  
5. **Sizing & Costs** → Bayesian win-rate + fractional Kelly; commission + spread modeled per trade.  
6. **Exits** → “Mean-touch”: close when z-score reverts to 0.  
7. **Metrics & Plots** → Equity & drawdown; per-trade stats.  
8. **Monte Carlo** → `Monte_carlo.py` bootstraps **trade returns** to simulate many equity paths.

---

## Files (what works with what)

- **Backtest/Backtest.py** – main backtest (signals, hedge, metrics).  
- **Backtest/Monte_carlo.py** – Monte-Carlo bootstrap of `pct_return` from the backtest.  
- **hypothesis_pca/adf.py** – stationarity (ADF) helpers.  
- **hypothesis_pca/granger.py** – Granger causality utilities.  
- **hypothesis_pca/pc3.py** – PCA factor provider used by the backtest.  
- **Results/summary.md** – short, readable summary.
