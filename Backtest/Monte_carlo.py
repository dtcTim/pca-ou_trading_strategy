from backtest_new_v3 import ou_backtest, gen_signaal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # kept for parity; not required

def monte_carlo():
    """Build signals and run OU backtest; return trade-level results."""
    windows = 50
    drempel_std = 2.75
    leverage = 5.0

    df = gen_signaal(drempel_std, windows)
    df_result, _, _, _, _, _, _, _, _ = ou_backtest(df, leverage)
    return df_result

def run():
    """Bootstrap trade returns to simulate equity paths vs. original backtest."""
    df = monte_carlo()

    n_sims = 10000
    n_trades = len(df)
    start_balans = 50_000

    for _ in range(n_sims):
        samples = np.random.choice(df["pct_return"], size=n_trades, replace=True)
        equity = start_balans * np.cumprod(1 + samples)
        plt.plot(equity, color="grey", alpha=0.1)

    equity_real = start_balans * np.cumprod(1 + df["pct_return"].to_numpy())
    plt.plot(equity_real, color="red", linewidth=2, label="Backtest")

    plt.title("Monte Carlo simulations vs Backtest")
    plt.xlabel("Trade number")
    plt.ylabel("Equity (€)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df

def main():
    """Entry point."""
    run()

if __name__ == "__main__":
    main()
