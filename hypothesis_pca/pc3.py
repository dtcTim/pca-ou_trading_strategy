import os
import pandas as pd
import numpy as np


# Paths
# -------------------------------------------------------------
# Base directory of this file
base_dir = os.path.dirname(os.path.abspath(__file__))
# Two directories up is assumed to contain the data folders
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))

# Timeframe key, folder path (keys kept as-is for compatibility)
path_dir = {
    '15min': os.path.join(parent_dir, 'data_15min'),
    '30min': os.path.join(parent_dir, 'data_30min'),
    'uur':   os.path.join(parent_dir, 'data_uur'),
    '4uur':  os.path.join(parent_dir, 'data_4uur'),
    'dag':   os.path.join(parent_dir, 'data_dag'),
    'week':  os.path.join(parent_dir, 'data_week')
}

def data_inladen(tijdframe, ticker):
    """Load a single parquet file for a ticker in a given timeframe.

    Expects a file named ``{ticker}.parquet`` containing at least a 'close' column.
    Returns a DataFrame with a DatetimeIndex and a single 'close' column.
    If the file is missing or invalid, an empty DataFrame is returned.
    """
    folder = path_dir.get(tijdframe)
    if folder is None:
        raise ValueError("Unknown timeframe '%s'" % tijdframe)

    fpath = os.path.join(folder, f"{ticker}.parquet")
    if not os.path.exists(fpath):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(fpath)
    except Exception:
        return pd.DataFrame()

    if 'close' not in df.columns:
        return pd.DataFrame()

    # Try to ensure a datetime index if a 'time' column exists
    if not isinstance(df.index, pd.DatetimeIndex) and 'time' in df.columns:
        df = df.set_index(pd.to_datetime(df['time']))

    return df.sort_index()[['close']]


# Build wide returns matrix (T x N)
# -------------------------------------------------------------
def svd_tickers(tijdframe, tickers, resample=None, min_frac=0.9):
    """Create a wide matrix of log-returns for the supplied tickers.

    Steps per ticker:
    - load parquet
    - optional resample (last price)
    - log return: log(close).diff()
    Rows must meet ``min_frac`` non-NaN coverage before final dropna.
    """
    cols = []
    for t in tickers:
        df = data_inladen(tijdframe, t)
        if df.empty:
            continue

        s = df['close']
        if resample:
            s = s.resample(resample).last()
        s = np.log(s).diff()              # log-returns
        cols.append(s.rename(t))

    if not cols:
        return pd.DataFrame()

    wide = pd.concat(cols, axis=1, join='outer').sort_index()

    # Keep only rows with enough data available across tickers
    keep = wide.notna().mean(axis=1) >= min_frac
    wide = wide[keep].dropna(how='any')
    return wide


# Standardization
# -------------------------------------------------------------
def z_score_globaal(df):
    """Column-wise z-score over the entire sample.

    This makes PCA/SVD act on the correlation structure rather than raw levels.
    Columns with zero variance are removed.
    """
    if df.empty:
        return df.copy()

    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=1)
    good = stds > 0

    Z = (df.loc[:, good] - means[good]) / stds[good]
    Z = Z.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    return Z


# SVD core
# -------------------------------------------------------------
def svd(df, pc_num=3):
    """Run economy-size SVD on standardized data and return (scores, loadings).

    df: standardized T x N matrix (rows=time, cols=tickers)
    pc_num: number of principal components to return

    scores (T x pc_num) are the PC time series (PC1..PCk)
    loadings (N x pc_num) are per-ticker correlations with each PC
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    A = df.to_numpy(dtype=float)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)  # A = U @ diag(S) @ Vt
    V = Vt.T
    k = int(max(1, min(pc_num, V.shape[1])))

    # Time-series scores
    scores = A @ V[:, :k]
    factors_df = pd.DataFrame(scores, index=df.index, columns=[f"PC{i+1}" for i in range(k)])

    # Approximate correlations (since input columns are z-scored)
    s_std = factors_df.std(axis=0, ddof=1).replace(0, np.nan)
    loadings = {}
    for j, pc in enumerate(factors_df.columns):
        s = factors_df.iloc[:, j].to_numpy(dtype=float)
        s_j = float(s_std.iloc[j])
        vals = {}
        for name in df.columns:
            x = df[name].to_numpy(dtype=float)
            cov = float(np.cov(x, s, ddof=1)[0, 1])
            vals[name] = cov / s_j if s_j and not np.isnan(s_j) else np.nan
        loadings[pc] = pd.Series(vals)

    loadings_df = pd.DataFrame(loadings).sort_index()
    return factors_df, loadings_df


# Public wrapper kept for compatibility with external callers
# -------------------------------------------------------------
def svd_pc3(timeframe, tickers, window=52, resample=None, min_frac=0.9):
    """Return PC1..PC3 score series for the given tickers.

    The ``window`` parameter is intentionally retained for backward compatibility
    with other modules that pass it, but is not used here.
    """
    R = svd_tickers(timeframe, tickers, resample=resample, min_frac=min_frac)
    Z = z_score_globaal(R)
    factors, _ = svd(Z, pc_num=3)
    return factors


# Example driver (optional for quick local runs)
# -------------------------------------------------------------
def pca(tf):
    """Run PC1..PC3 for a default FX universe ."""
    tickers = [
        "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD",
        "CADCHF", "CADJPY",
        "CHFJPY",
        "EURAUD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURNZD", "EURUSD",
        "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY", "GBPNZD", "GBPUSD",
        "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
        "USDCAD", "USDCHF", "USDJPY",
    ]
    return svd_pc3(tf, tickers)


def main():
    df = pca('dag')
    print(df.head())


if __name__ == '__main__':
    main()
