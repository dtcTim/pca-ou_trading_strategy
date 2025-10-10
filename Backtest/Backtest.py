"""
Hedged OU backtest with factor hedging (PCs), Bayesian+Kelly sizing, and a basic cost model.
- Public version: no strategically sensitive pair selections or tunings hardcoded.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import beta as beta_dist

# External module delivering your factor series; keep the name unchanged for compatibility.
from pc3 import pca

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    filename="grid_single_test_v2.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ------------------------------------------------------------
# Path config (unchanged)
# ------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
path_dir = {
    "15min": os.path.join(parent_dir, "data_15min"),
    "30min": os.path.join(parent_dir, "data_30min"),
    "uur": os.path.join(parent_dir, "data_uur"),
    "4uur": os.path.join(parent_dir, "data_4uur"),
    "dag": os.path.join(parent_dir, "data_dag"),
    "week": os.path.join(parent_dir, "data_week"),
}

def data_inladen(tijdframe: str, ticker: str) -> pd.DataFrame:
    """Load OHLC parquet data for (tijdframe, ticker)."""
    if tijdframe not in path_dir:
        print(f"opgegeven tijdframe is niet juist: {tijdframe}")
        return False
    tf_pad = path_dir[tijdframe]
    ticker_pad = os.path.join(tf_pad, f"{ticker}.parquet")
    if os.path.exists(ticker_pad):
        df = pd.read_parquet(ticker_pad)
        if df.empty:
            print(f"df is leeg voor: {ticker}")
        return df
    else:
        print(f"geen data beschikbaar voor {ticker}")
        return pd.DataFrame()

# Universe (non-sensitive; major/minor FX codes only)
POOL = [
    "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
    "AUDCAD","AUDCHF","AUDJPY","AUDNZD","CADCHF","CADJPY","CHFJPY",
    "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD",
    "GBPAUD","GBPCAD","GBPCHF","GBPJPY","GBPNZD",
    "NZDCAD","NZDCHF","NZDJPY"
]

# === Legacy compatibility toggle for the v1 curve
V1_COMPAT = False  # raw ΔPC hedge, mean-touch exits, zero costs

# Factors to hedge (kept generic for public repo)
FACTORS = ("PC3", "PC2", "PC1")

# Ridge-hedge settings (not sensitive; needed for numerical stability)
RIDGE_WINDOW = 100 #(random)
RIDGE_LAMBDA = 5e-4

# Bayesian prior (not sensitive)
BETA_PRIOR_ALPHA = 0.5
BETA_PRIOR_BETA = 0.5
POST_Q_LB = 0.0  # p_use = mean

# Kelly sizing caps — defaults preserved for compatibility
FRACTION = 2.0 
FLOOR = 1.25
CAP = 8

EPS = 1e-6

def load_pool_returns(tf: str, exclude_ticker: str | None = None) -> pd.DataFrame:
    """Log returns of the pool (excluding the target ticker)."""
    out = {}
    for t in POOL:
        if t == exclude_ticker:
            continue
        dfp = data_inladen(tf, t)
        if dfp is None or dfp.empty or "close" not in dfp:
            continue
        out[t] = np.log(dfp["close"]).diff().rename(t)
    if not out:
        return pd.DataFrame()
    return pd.concat(out.values(), axis=1).sort_index().dropna(how="all")

def rolling_ridge_weights(dF: pd.Series, R_pool: pd.DataFrame, window: int = 250, lam: float = 1e-3):
    """Rolling ridge regression to replicate ΔF using pool returns."""
    idx = R_pool.index.intersection(dF.index)
    R = R_pool.loc[idx].fillna(0.0)
    y = dF.loc[idx].fillna(0.0)
    cols = R.columns
    W = pd.DataFrame(index=idx, columns=cols, dtype=float)
    I = np.eye(R.shape[1])

    for i in range(len(idx)):
        if i < window:
            continue
        sl = slice(i - window, i)
        X = R.iloc[sl].values
        yy = y.iloc[sl].values.reshape(-1, 1)
        XtX = X.T @ X + lam * I
        Xty = X.T @ yy
        try:
            gamma = np.linalg.solve(XtX, Xty).flatten()
        except np.linalg.LinAlgError:
            gamma = np.zeros(R.shape[1])
        W.iloc[i] = gamma

    dF_hat = (R * W.shift(1)).sum(axis=1)  # weights(t-1) applied to R(t)
    return W, dF_hat.reindex(idx)

def ou_backtest(df: pd.DataFrame, leverage: float = 1.0):
    """Vectorized backtest loop with mean-touch exits and factor-hedged PnL."""
    # Cost model (illustrative; configurable; not sensitive)
    start_balans = 50_000.0
    commission_pct = 0.0002
    spread_pct = 0.0001

    balans = start_balans
    trades, positions = [], {}
    bayes_state = {"wins": 0, "losses": 0, "sum_win": 0.0, "sum_loss": 0.0}
    laatste_prijs = {}
    laatste_cdf = {fac: {} for fac in FACTORS}

    n_pairs = df["pair"].nunique()
    balans_per_pair = start_balans / n_pairs * leverage

    df = df.sort_values("pair").sort_index()
    df["vorige_close"] = df.groupby("pair")["close"].shift(1)

    daily_equity = {}
    current_date = None  # reserved for future use

    for index, row in df.iterrows():
        ticker = row["ticker"]
        pair = row["pair"]
        tf = row["tf"]
        prijs = row["close"]
        vorige_prijs = row["vorige_close"]
        long_signaal = row["long_signaal"]
        short_signaal = row["short_signaal"]

        bar_date = index.date()
        laatste_prijs[pair] = prijs
        for fac in FACTORS:
            cdf_col = f"cdf_{fac}"
            if cdf_col in row and pd.notna(row[cdf_col]):
                laatste_cdf[fac][pair] = float(row[cdf_col])

        # Per-bar hedged PnL for open positions
        bar_hedged_returns = []
        for pos_pair, pos_info in positions.items():
            if pos_info is None or pos_pair != pair:
                continue
            if pd.notna(vorige_prijs):
                dy = np.log(prijs / vorige_prijs)
                hedge_shift = 0.0
                for fac in FACTORS:
                    beta_fac = pos_info["beta_entry"].get(fac, 0.0)
                    dF_col = f"dFhat_{fac}"
                    if dF_col in row and pd.notna(row[dF_col]):
                        dF = float(row[dF_col])
                        hedge_shift += beta_fac * dF
                hedged_ret = dy - hedge_shift
                side_sign = pos_info["side_sign"]
                notional = pos_info["notional"]
                bar_pnl = side_sign * notional * hedged_ret
                bar_hedged_returns.append(bar_pnl)
                pos_info["current_pnl"] = pos_info.get("current_pnl", 0.0) + bar_pnl

        total_bar_pnl = sum(bar_hedged_returns)
        balans += total_bar_pnl
        daily_equity[bar_date] = balans

        # Exit logic: mean-touch using current z-score
        positions_to_close = []
        for pos_pair, pos_info in positions.items():
            if pos_info is None or pos_pair != pair:
                continue
            z_current = row.get("z_score", np.nan)
            position_type = pos_info["type"]
            if position_type == "long":
                exit_cond = (pd.notna(z_current) and z_current >= 0.0)
            else:
                exit_cond = (pd.notna(z_current) and z_current <= 0.0)
            if exit_cond:
                notional = pos_info["notional"]
                exit_costs = notional * (commission_pct + spread_pct)
                balans -= exit_costs
                daily_equity[bar_date] = balans
                net_pnl = pos_info.get("current_pnl", 0.0) - exit_costs - pos_info.get("entry_costs", 0.0)
                if net_pnl > 0:
                    bayes_state["wins"] += 1
                    bayes_state["sum_win"] += net_pnl
                elif net_pnl < 0:
                    bayes_state["losses"] += 1
                    bayes_state["sum_loss"] += abs(net_pnl)
                trades.append({
                    "pair": pos_pair,
                    "position_type": position_type,
                    "balans": balans,
                    "entry_date": pos_info["entry_date"],
                    "exit_date": index,
                    "entry_price": pos_info["entry_price"],
                    "exit_price": prijs,
                    "gross_pnl": pos_info.get("current_pnl", 0.0),
                    "entry_costs": pos_info.get("entry_costs", 0.0),
                    "exit_costs": exit_costs,
                    "net_pnl": net_pnl,
                    "notional": notional,
                    "leverage_used": leverage,
                    "exit_reason": "mean_touch",
                    "tf": tf,
                })
                positions_to_close.append(pos_pair)

        for p_close in positions_to_close:
            positions[p_close] = None

        # Entry logic
        if pair not in positions:
            positions[pair] = None
        if (long_signaal or short_signaal) and positions[pair] is None:
            if pd.isna(vorige_prijs):
                continue

            # Global Bayesian + (fractional) Kelly sizing
            w, l = bayes_state["wins"], bayes_state["losses"]
            a_post = BETA_PRIOR_ALPHA + w
            b_post = BETA_PRIOR_BETA + l
            p_lb = beta_dist.ppf(POST_Q_LB, a_post, b_post)
            p_mean = a_post / max(a_post + b_post, 1e-9)
            p_use = p_mean if (w + l) >= 100 else (0.5 if not np.isfinite(p_lb) else p_lb)

            avg_win = (bayes_state["sum_win"] / w) if w > 0 else 0.0
            avg_loss = (bayes_state["sum_loss"] / l) if l > 0 else 0.0
            b_ratio = np.clip(avg_win / max(avg_loss, EPS), EPS, 100.0)

            kelly_frac = ((b_ratio + 1.0) * p_use - 1.0) / b_ratio
            kelly_frac = float(np.clip(kelly_frac, 0.0, 1.0))
            sizing_factor = np.clip(FRACTION * kelly_frac, FLOOR, CAP)

            notional = balans_per_pair * sizing_factor

            entry_costs = notional * (commission_pct + spread_pct)
            if balans < entry_costs:
                continue
            balans -= entry_costs
            daily_equity[bar_date] = balans

            if long_signaal:
                position_type = "long"
                side_sign = +1
            else:
                position_type = "short"
                side_sign = -1

            beta_entry = {fac: float(row.get(f"beta_{fac}", 0.0)) for fac in FACTORS}
            cdf_entry = {fac: float(row.get(f"cdf_{fac}", 0.0)) for fac in FACTORS}

            positions[pair] = {
                "type": position_type,
                "entry_date": index,
                "entry_price": vorige_prijs,
                "side_sign": side_sign,
                "notional": notional,
                "beta_entry": beta_entry,
                "cdf_entry": cdf_entry,
                "current_pnl": 0.0,
                "entry_costs": entry_costs,
                "ticker": ticker,
                "tf": tf,
            }

    # Force-exit remaining positions at final timestamp
    final_timestamp = df.index[-1]
    final_date = final_timestamp.date()
    for pair_key, pos_info in positions.items():
        if pos_info is None:
            continue
        exit_prijs = laatste_prijs.get(pair_key, pos_info["entry_price"])
        notional = pos_info["notional"]
        exit_costs = notional * (commission_pct + spread_pct)
        balans -= exit_costs
        net_pnl = pos_info.get("current_pnl", 0.0) - exit_costs - pos_info.get("entry_costs", 0.0)
        trades.append({
            "pair": pair_key,
            "ticker": pos_info.get("ticker", ""),
            "timeframe": pos_info.get("tf", ""),
            "position_type": pos_info["type"],
            "balans": balans,
            "entry_date": pos_info["entry_date"],
            "exit_date": final_timestamp,
            "entry_price": pos_info["entry_price"],
            "exit_price": exit_prijs,
            "gross_pnl": pos_info.get("current_pnl", 0.0),
            "entry_costs": pos_info.get("entry_costs", 0.0),
            "exit_costs": exit_costs,
            "net_pnl": net_pnl,
            "notional": notional,
            "leverage_used": leverage,
            "exit_reason": "EOD",
            "tf": tf,
        })

    daily_equity[final_date] = balans

    # --------------------------------------------------------
    # Equity/Drawdown stats (required for plots & logging)
    # --------------------------------------------------------
    if daily_equity:
        df_daily = pd.DataFrame.from_dict(daily_equity, orient="index", columns=["balans"])
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily = df_daily.sort_index()
        df_daily["running_max"] = df_daily["balans"].expanding().max()
        df_daily["drawdown_pct"] = (df_daily["balans"] - df_daily["running_max"]) / df_daily["running_max"] * 100
        max_drawdown = df_daily["drawdown_pct"].min()
        negative_dd = df_daily["drawdown_pct"][df_daily["drawdown_pct"] < 0]
        avg_drawdown = negative_dd.mean() if len(negative_dd) > 0 else 0.0
        median_drawdown = negative_dd.median() if len(negative_dd) > 0 else 0.0
        percentile_95_dd = negative_dd.quantile(0.05) if len(negative_dd) > 0 else 0.0
        dagen_in_drawdown = (df_daily["drawdown_pct"] < -0.1).sum()
        pct_dagen_in_dd = (dagen_in_drawdown / len(df_daily)) * 100

        # Episode length (indicative)
        dd_episodes, in_drawdown, start_dd = [], False, None
        for date, rowd in df_daily.iterrows():
            if rowd["drawdown_pct"] < -0.1 and not in_drawdown:
                in_drawdown, start_dd = True, date
            elif rowd["drawdown_pct"] >= -0.1 and in_drawdown:
                in_drawdown = False
                if start_dd is not None:
                    dd_episodes.append((date - start_dd).days)
        avg_dd_duration = np.mean(dd_episodes) if dd_episodes else 0
    else:
        max_drawdown = avg_drawdown = median_drawdown = percentile_95_dd = pct_dagen_in_dd = avg_dd_duration = 0
        df_daily = pd.DataFrame()

    df_trades = pd.DataFrame(trades)
    n_trades = len(df_trades)
    if n_trades == 0:
        return None, 0, 0, 0, 0, 0, 0, 0, 0

    total_pnl = balans - start_balans
    win_rate = (df_trades["net_pnl"] > 0).mean() if n_trades > 0 else 0
    rendement = (balans - start_balans) / start_balans * 100
    df_trades["pct_return"] = df_trades["net_pnl"] / df_trades["notional"]

    # Sharpe computed from equity curve
    if len(df_daily) > 1:
        df_daily["daily_return"] = df_daily["balans"].pct_change()
        daily_returns = df_daily["daily_return"].dropna()
        if len(daily_returns) > 1:
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std(ddof=1)
            sharpe_ratio = (mean_daily_return * 252) / (std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0

    if df_trades["entry_date"].notna().any():
        eerste_trade_datum = df_trades["entry_date"].dropna().min()
        laatste_trade_datum = df_trades["exit_date"].dropna().max()
        dagen_gehandeld = (laatste_trade_datum - eerste_trade_datum).days
        jaren_gehandeld = max(dagen_gehandeld / 365, 1 / 365)
        groeifactor = balans / start_balans if start_balans > 0 else 1
        jaarlijks_rendement = ((groeifactor ** (1 / jaren_gehandeld)) - 1) * 100 if groeifactor > 0 else -100
    else:
        jaarlijks_rendement = 0

    winning_trades = df_trades[df_trades["net_pnl"] > 0]
    losing_trades = df_trades[df_trades["net_pnl"] < 0]
    avg_win = winning_trades["net_pnl"].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades["net_pnl"].mean() if len(losing_trades) > 0 else 0
    gross_profit = winning_trades["net_pnl"].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades["net_pnl"].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # --------------------------------------------------------
    # OPTIONAL: Visualization 
    # --------------------------------------------------------
    if not df_daily.empty:
        plt.figure(figsize=(15, 10))

        # Subplot 1: Equity curve
        plt.subplot(2, 1, 1)
        plt.plot(df_daily.index, df_daily['balans'], linewidth=2, label='Portfolio Equity')
        plt.plot(df_daily.index, df_daily['running_max'], '--', alpha=0.7, label='Running Maximum')
        plt.title('Hedged OU Portfolio - Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (€)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(df_daily.index, df_daily['drawdown_pct'], 0, alpha=0.3, color='red', label='Drawdown')
        plt.plot(df_daily.index, df_daily['drawdown_pct'], linewidth=1)
        plt.title('Daily Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.show()

    # --------------------------------------------------------
    # OPTIONAL: Logging summary
    # --------------------------------------------------------
    if n_trades > 0:
        long_trades = df_trades[df_trades['position_type'] == 'long']
        short_trades = df_trades[df_trades['position_type'] == 'short']

        logging.info("=" * 60)
        logging.info("BACKTEST RESULTATEN")
        logging.info("=" * 60)
        logging.info(f"Periode:                 {df.index[0].strftime('%Y-%m-%d')} tot {df.index[-1].strftime('%Y-%m-%d')}")
        logging.info(f"Hefboom:                 {leverage:.1f}x")
        logging.info(f"Commissie:               {commission_pct:.4f} ({commission_pct*100:.2f}%)")
        logging.info(f"Spread:                  {spread_pct:.4f} ({spread_pct*100:.2f}%)")
        logging.info("-" * 60)
        logging.info(f"Start balans:            €{start_balans:,.2f}")
        logging.info(f"Eind balans:             €{balans:,.2f}")
        logging.info(f"Totaal PnL:              €{total_pnl:,.2f}")
        logging.info(f"Totale kosten:           €{(df_trades['entry_costs'].sum()+df_trades['exit_costs'].sum()):,.2f}")
        logging.info(f"  - Entry kosten:        €{df_trades['entry_costs'].sum():,.2f}")
        logging.info(f"  - Exit  kosten:        €{df_trades['exit_costs'].sum():,.2f}")
        logging.info("-" * 60)
        logging.info(f"Totaal rendement:        {rendement:.2f}%")
        logging.info(f"Jaarlijks rendement:     {jaarlijks_rendement:.2f}%")
        logging.info(f"Sharpe ratio:            {sharpe_ratio:.3f}")
        logging.info(f"Maximum drawdown:        {max_drawdown:.2f}%")
        logging.info(f"Gemiddelde drawdown:     {avg_drawdown:.2f}%")
        logging.info(f"Mediaan drawdown:        {median_drawdown:.2f}%")
        logging.info(f"95e percentiel DD:       {percentile_95_dd:.2f}%")
        logging.info(f"Dagen in drawdown:       {pct_dagen_in_dd:.1f}%")
        logging.info(f"Gem. DD duur:            {avg_dd_duration:.1f} dagen")
        logging.info(f"Profit factor:           {profit_factor:.2f}")
        logging.info("-" * 60)
        logging.info(f"Aantal trades:           {n_trades}")
        logging.info(f"  - Long trades:         {len(long_trades)}")
        logging.info(f"  - Short trades:        {len(short_trades)}")
        logging.info(f"Win rate:                {win_rate:.2%}")
        logging.info(f"Gem. winnende trade:     €{(df_trades[df_trades['net_pnl']>0]['net_pnl'].mean() if (df_trades['net_pnl']>0).any() else 0):.2f}")
        logging.info(f"Gem. verliezende trade:  €{(df_trades[df_trades['net_pnl']<0]['net_pnl'].mean() if (df_trades['net_pnl']<0).any() else 0):.2f}")
        logging.info(f"Dagelijkse equity punten: {len(df_daily)}")
        logging.info("=" * 60)
        logging.info("")

    return (
        df_trades,
        n_trades,
        total_pnl,
        win_rate,
        rendement,
        jaarlijks_rendement,
        sharpe_ratio,
        max_drawdown,
        profit_factor,
    )

def ou(timeframe: str, ticker: str, lag: int, window: int, drempel_std: float) -> pd.DataFrame | None:
    """Build OU spread vs. factor lags and generate signals + hedge inputs."""
    df = data_inladen(timeframe, ticker)
    if df is None or df.empty:
        return None
    y = np.log(df["close"]).rename("y").dropna()

    df_pc = pca(timeframe)  # expects columns like 'PC1','PC2','PC3', ...
    if df_pc is None or df_pc.empty:
        return None

    # Design matrix with lags of the selected factors
    Xcols = []
    X = pd.DataFrame(index=y.index)
    for fac in FACTORS:
        if fac not in df_pc.columns:
            continue
        F = df_pc[fac].dropna().cumsum().rename(fac)  # factor level
        Fl = F.shift(lag).rename(f"{fac}_lag")
        X = X.join(Fl, how="inner")
        Xcols.append(f"{fac}_lag")

    dfn = pd.concat([y, X], axis=1).dropna()
    if len(Xcols) == 0 or dfn.empty:
        return None

    X_ = sm.add_constant(dfn[Xcols])
    ols = sm.OLS(dfn["y"], X_).fit()

    # Residual + z-score
    z = (dfn["y"] - ols.predict(X_)).rename("z")
    out = pd.concat([dfn[["y"]], z], axis=1)
    out["z_mu"] = out["z"].rolling(window).mean()
    out["z_sigma"] = out["z"].rolling(window).std(ddof=0)
    out["z_score"] = (out["z"] - out["z_mu"]) / out["z_sigma"]

    # Factor replica dFhat per factor (for hedge shift)
    R_pool = load_pool_returns(timeframe, exclude_ticker=ticker)
    for fac in FACTORS:
        if fac not in df_pc.columns:
            continue
        F = df_pc[fac].dropna().cumsum()
        dF = F.diff().rename(f"dF_{fac}")
        if V1_COMPAT:
            dF_series = dF
        else:
            if R_pool.empty:
                dF_series = pd.Series(0.0, index=out.index)
            else:
                _, dF_hat = rolling_ridge_weights(dF, R_pool, window=RIDGE_WINDOW, lam=RIDGE_LAMBDA)
                dF_series = dF_hat
        out[f"dFhat_{fac}"] = dF_series.reindex(out.index).astype(float)
        out[f"cdf_{fac}"] = out[f"dFhat_{fac}"].fillna(0.0).cumsum()

    # Betas (stable by name, independent of index)
    for fac in FACTORS:
        parnaam = f"{fac}_lag"
        beta_val = float(ols.params.get(parnaam, 0.0)) if hasattr(ols, "params") else 0.0
        out[f"beta_{fac}"] = beta_val

    # Signals + metadata
    out = out.join(df[["close", "high", "low"]], how="inner")
    out["ticker"] = ticker
    out["tf"] = timeframe
    out["pair"] = out["ticker"] + "@" + out["tf"]
    out["long_signaal"] = out["z_score"] <= -drempel_std
    out["short_signaal"] = out["z_score"] >= drempel_std
    return out.dropna(subset=["z_sigma"])

def gen_signaal(drempel_std: float, window: int) -> pd.DataFrame:
    """
    Generate features/signals for all (tf, ticker, lag) combinations.

     FORMAT:
     pairs = [
         # (timeframe, ticker, lag)
         # Valid timeframes: "15min","30min","uur","4uur","dag","week"
         # Example entries (replace with your own):
         # ("uur",  "EURUSD", 2),
         # ("4uur", "USDJPY", 4),
         # ("dag",  "AUDNZD", 1),
    ]
    """
    # Define your curated selection here. Keep this list private in your repo if needed.
    pairs: list[tuple[str, str, int]] = [
        # (timeframe, ticker, lag),
        # e.g. ("uur", "EURUSD", 2),
        # e.g. ("4uur", "USDJPY", 4),
        # e.g. ("dag", "AUDNZD", 1),
    ]

    frames = []
    for tf, ticker, lag in pairs:
        df = ou(tf, ticker, lag, window, drempel_std)
        if df is None:
            continue
        f = df.sort_index()
        # ATR for context/risk sizing downstream (no lookahead)
        f["prev_close"] = f.groupby("pair")["close"].shift(1)
        f["tr1"] = f["high"] - f["low"]
        f["tr2"] = (f["high"] - f["prev_close"]).abs()
        f["tr3"] = (f["low"] - f["prev_close"]).abs()
        f["TR"] = f[["tr1", "tr2", "tr3"]].max(axis=1)
        f["ATR"] = f.groupby("pair")["TR"].transform(lambda s: s.rolling(window=14).mean())
        frames.append(f)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0).sort_values(["pair"]).sort_index()

def run_een_combinatie(params: tuple):
    """Helper to run one (window, drempel_std, leverage) combination."""
    window, drempel_std, leverage = params
    try:
        df = gen_signaal(drempel_std, window)
        results = ou_backtest(df, leverage)

        if len(results) >= 9:
            (df_trades, n_trades, total_pnl, win_rate, rendement,
             jaarlijks_rendement, sharpe_ratio, max_drawdown, profit_factor) = results
        else:  # legacy fallback
            df_trades, n_trades, total_pnl, win_rate, rendement, jaarlijks_rendement = results
            sharpe_ratio = max_drawdown = profit_factor = 0

        if isinstance(df_trades, pd.DataFrame) and not df_trades.empty:
            by_pair = df_trades.groupby(["pair", "timeframe"])["net_pnl"].agg(["count", "sum"]).sort_values("sum", ascending=False)
            print("-" * 50)
            print(f"\nper paar resultaat: {by_pair}")
            print("-" * 50)

        return {
            "window": window,
            "drempel_std": drempel_std,
            "leverage": leverage,
            "n_trades": n_trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "rendement": rendement,
            "jaarlijks_rendement": jaarlijks_rendement,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
        }
    except Exception as e:
        logging.error(f"Error in combinatie {params}: {e}")
        return {
            "window": window,
            "drempel_std": drempel_std,
            "leverage": leverage,
            "n_trades": 0,
            "total_pnl": 0,
            "win_rate": 0,
            "rendement": -999,
            "jaarlijks_rendement": -999,
            "sharpe_ratio": -999,
            "max_drawdown": 999,
            "profit_factor": 0
        }

def run() -> pd.DataFrame:
    """Entry point: run a single combination and print/log results."""
    # Public, non-sensitive placeholders.
    windows = 100
    drempel_std = 2
    leverage = 1.0

    params = (windows, drempel_std, leverage)
    result = run_een_combinatie(params)
    df_results = pd.DataFrame([result])
    df_results = df_results[df_results["jaarlijks_rendement"] > -999]

    if len(df_results) > 0:
        df_results = df_results.sort_values("sharpe_ratio", ascending=False)

        print("\n" + "=" * 100)
        print("BACKTEST RESULTATEN MET HEFBOOM EN KOSTEN:")
        print("=" * 100)
        for i, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
            line_console = (
                f"{i}. Sharpe: {row['sharpe_ratio']:.3f} | "
                f"Jaarlijks rendement: {row['jaarlijks_rendement']:.2f}% | "
                f"Max DD: {row['max_drawdown']:.2f}% | "
                f"PF: {row['profit_factor']:.2f} | "
                f"Trades: {row['n_trades']} | Win Rate: {row['win_rate']:.2%} | "
                f"Leverage: {row['leverage']:.1f}x | "
            )
            print(line_console)
            logging.info(line_console)

        print("\n" + "=" * 100)
        print("PARAMETER DETAILS:")
        print("=" * 100)
        best_row = df_results.iloc[0]
        print(f"Window: {best_row['window']}")
        print(f"Drempel std: {best_row['drempel_std']}")
        print(f"Leverage: {best_row['leverage']:.1f}x")
        print("=" * 100)
    else:
        print("Geen geldige resultaten gevonden (voeg entries toe in 'pairs' binnen gen_signaal()).")

    return df_results

if __name__ == "__main__":
    df_results = run()
