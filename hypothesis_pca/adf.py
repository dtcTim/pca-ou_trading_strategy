import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Paths
# -----------------------------------------------------------------------------
# Base directory of this file
base_dir = os.path.dirname(os.path.abspath(__file__))
# Two directories up is assumed to contain the data folders
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))

# Timeframe key -> folder path (keys kept as-is for compatibility)
path_dir = {
    '15min': os.path.join(parent_dir, 'data_15min'),
    '30min': os.path.join(parent_dir, 'data_30min'),
    'uur':   os.path.join(parent_dir, 'data_uur'),
    '4uur':  os.path.join(parent_dir, 'data_4uur'),
    'dag':   os.path.join(parent_dir, 'data_dag'),
    'week':  os.path.join(parent_dir, 'data_week'),
}


# IO helper (kept name for backward compatibility)
# -----------------------------------------------------------------------------
def data_inladen(tijdframe, ticker):
    """Load a single parquet file for a ticker in a given timeframe.

    Expects a file named ``{ticker}.parquet`` with at least a 'close' column.
    Returns a DataFrame with a DatetimeIndex and a single 'close' column.
    On missing/invalid data, returns an empty DataFrame.
    """
    folder = path_dir.get(tijdframe)
    if folder is None:
        return pd.DataFrame()

    fpath = os.path.join(folder, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(fpath)
    except Exception:
        return pd.DataFrame()

    if 'close' not in df.columns:
        return pd.DataFrame()

    # Ensure datetime index when a 'time' column is present
    if not isinstance(df.index, pd.DatetimeIndex) and 'time' in df.columns:
        df = df.set_index(pd.to_datetime(df['time']))

    return df.sort_index()[['close']]


# Spread construction and OU/ADF diagnostics
# -----------------------------------------------------------------------------
# Note: this module is import-only (no side effects / no CLI). The interface is
# intentionally minimal for recruiters: call spread(...) to construct the
# residual, and ou(...) to obtain stationarity and mean-reversion diagnostics.

from pc3 import pca  # uses the existing pca(timeframe) DataFrame with PC1..PC3


def spread(timeframe, ticker, lag):
    """Build a linear spread z_t between log-price and lagged cumulative PC3.

    Steps
    -----
    1) y_t = log(close) of the selected ticker.
    2) Compute PC3 from pca(timeframe), then F_t = cumsum(PC3).
    3) Regress y_t on const + F_{t-lag}. Residual is z_t.

    Returns
    -------
    out : DataFrame
        Columns: ['y', 'F_lag', 'z'] aligned on the common index.
    ols : RegressionResults
        statsmodels OLS fit of y on [const, F_lag].
    """
    df = data_inladen(timeframe, ticker)
    if df.empty:
        return pd.DataFrame(), None

    y = np.log(df['close']).rename('y').dropna()

    df_pc3 = pca(timeframe)  # expects columns PC1..PC3
    if 'PC3' not in df_pc3.columns:
        return pd.DataFrame(), None

    pc3 = df_pc3['PC3'].dropna()

    # Cumulative PC3 as a level-like factor (integration over time)
    F = pc3.cumsum().rename('F')
    F_lag = F.shift(lag).rename('F_lag')

    dfn = pd.concat([y, F_lag], axis=1).dropna()

    # Add constant term so the line is not forced through (0, 0)
    X = sm.add_constant(dfn['F_lag'])
    ols = sm.OLS(dfn['y'], X).fit()

    z = (dfn['y'] - ols.predict(X)).rename('z')
    out = pd.concat([dfn['y'], dfn['F_lag'], z], axis=1).dropna()
    return out, ols


def ou(timeframe, ticker, lag):
    """Compute OU-style mean-reversion diagnostics on the spread residual z_t.

    Metrics
    -------
    - adf_p: p-value of the ADF test on z_t (H0: unit root). Lower is better.
    - phi:   AR(1) coefficient from OLS(z_t ~ const + z_{t-1}).
    - kappa: OU speed mapped as -log(phi) per bar (if phi > 0).
    - half_life_bars: log(2)/kappa (in bars) if kappa > 0.
    - p_phi_lt1: one-sided t-test P(phi < 1) based on OLS standard error.
    - n_obs: number of observations used in AR(1).

    Returns
    -------
    dict
        A dictionary with the fields above. Values may be NaN when not defined.
    """
    dfn, ols = spread(timeframe, ticker, lag)
    if dfn is None or dfn.empty:
        return {
            'adf_p': np.nan, 'phi': np.nan, 'kappa': np.nan,
            'half_life_bars': np.nan, 'p_phi_lt1': np.nan, 'n_obs': 0,
        }

    z = dfn['z'].dropna().astype(float)

    # ADF p-value for stationarity of z_t
    adf_stat, p_adf, *_ = adfuller(z, regression='c', autolag='AIC')

    # AR(1): z_t = alpha + phi * z_{t-1} + eps_t
    z_lag = z.shift(1).dropna()
    z_t = z.loc[z_lag.index]
    X = sm.add_constant(z_lag)
    ar1 = sm.OLS(z_t, X).fit()

    # Extract phi and its std error
    phi = float(ar1.params.get(z_lag.name, np.nan))
    se_phi = float(ar1.bse.get(z_lag.name, np.nan))

    # Map to OU parameters (per bar)
    kappa = -np.log(phi) if (isinstance(phi, float) and phi > 0) else np.nan
    half_life = (np.log(2) / kappa) if (isinstance(kappa, float) and np.isfinite(kappa) and kappa > 0) else np.nan

    # One-sided t-test for H0: phi = 1 vs H1: phi < 1
    try:
        from scipy.stats import t as t_dist
        if np.isfinite(se_phi) and se_phi > 0:
            tstat = (phi - 1.0) / se_phi
            dfree = int(ar1.df_resid)
            p_phi_lt1 = float(t_dist.cdf(tstat, dfree))
        else:
            p_phi_lt1 = np.nan
    except Exception:
        p_phi_lt1 = np.nan

    return {
        'adf_p': float(p_adf),
        'phi': float(phi) if np.isfinite(phi) else np.nan,
        'kappa': float(kappa) if np.isfinite(kappa) else np.nan,
        'half_life_bars': float(half_life) if np.isfinite(half_life) else np.nan,
        'p_phi_lt1': p_phi_lt1,
        'n_obs': int(len(z_t)),
    }


# Note: no executable code (no main). This module is safe to import in any context.
