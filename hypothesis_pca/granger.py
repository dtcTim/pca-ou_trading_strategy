import pandas as pd
import numpy as np
from pc3 import pca
import os
from statsmodels.tsa.stattools import grangercausalitytests
import logging

# Hypothesis testing framework:
# H0: PC3 has no predictive power for FX returns
# H1: PC3 has predictive power for FX returns

logging.basicConfig(filename=f'res_granger.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory structure for different timeframes (data stored in parquet files)
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir, os.pardir))
path_dir = {
    '15min': os.path.join(parent_dir, 'data_15min'),
    '30min': os.path.join(parent_dir, 'data_30min'),
    'uur':   os.path.join(parent_dir, 'data_uur'),
    '4uur':  os.path.join(parent_dir, 'data_4uur'),
    'dag':   os.path.join(parent_dir, 'data_dag'),
    'week':  os.path.join(parent_dir, 'data_week')
}

def data_inladen(tijdframe, ticker):
    """
    Loads FX data from parquet files for a given timeframe and ticker.
    Returns a pandas DataFrame or an empty DataFrame if not available.
    """
    if tijdframe not in path_dir:
        logging.info(f'Invalid timeframe provided: {tijdframe}')
        return False
    tf_pad = path_dir[tijdframe]
    ticker_pad = os.path.join(tf_pad, f'{ticker}.parquet')
    if os.path.exists(ticker_pad):
        df = pd.read_parquet(ticker_pad)
        if df.empty:
            logging.info(f'DataFrame is empty for: {ticker}')
        return df
    else:
        logging.info(f'No data available for {ticker}')
        return pd.DataFrame()

def functie(timeframe):
    """
    For a given timeframe:
    1. Load PC3 factor (via custom PCA function 'pca').
    2. Load returns for all major FX pairs.
    3. Compute correlation between PC3 and returns.
    4. Return a ranking of tickers by absolute correlation.
    """
    tickers = [
        "EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
        "AUDCAD","AUDCHF","AUDJPY","AUDNZD","CADCHF","CADJPY","CHFJPY",
        "EURAUD","EURCAD","EURCHF","EURGBP","EURJPY","EURNZD","GBPAUD",
        "GBPCAD","GBPCHF","GBPJPY","GBPNZD","NZDCAD","NZDCHF","NZDJPY"
    ]

    df_pc3 = pca(timeframe)
    pc3 = df_pc3['PC3'].dropna() 
    correlaties = {}

    for ticker in tickers:
        df = data_inladen(timeframe, ticker)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        returns = df['returns'].dropna()

        # Align PC3 with returns by timestamp
        merge = pd.merge(pc3.to_frame('PC3'), returns.to_frame('returns'),
                         left_index=True, right_index=True, how='inner')

        if len(merge) < 50:
            logging.info(f'Not enough data points: {len(merge)}')

        # Pearson correlation
        corr = merge['returns'].corr(merge['PC3'])
        if not np.isnan(corr):
            correlaties[ticker] = abs(corr)
            logging.info(f"{ticker}: correlation = {corr:.4f} (abs: {abs(corr):.4f})")

    beste_correlaties = sorted(correlaties.items(), key=lambda x: x[1], reverse=True)
    return beste_correlaties

def grangers(timeframe, lag):
    """
    Perform Granger causality tests:
    1. Select top 5 FX pairs with the highest correlation to PC3.
    2. Run Granger test to check if PC3 has predictive power on returns.
    3. Log significant results (p < 0.05).
    """
    best_results = functie(timeframe)
    best_corr_tickers = [t for t, _ in best_results[:5]]

    df_pc3 = pc3(timeframe)
    pc3 = df_pc3['PC2'].dropna()

    gc_list = {}
    for item in best_corr_tickers:
        df = data_inladen(timeframe, item)
        df['returns'] = np.log(df['close'] / df['close'].shift(1))

        merge = pd.merge(df, pc3.to_frame('PC2'),left_index=True, right_index=True, how='inner')
        data_stat = merge[['returns', 'PC2']].dropna()

        logging.info(f'Dataset size for Granger test: {len(data_stat)}')
        gc_res = grangercausalitytests(data_stat, lag)
        gc_list[item] = gc_res

        # Log significance for each lag
        for k in range(1, lag + 1):
            p = gc_res[k][0]['ssr_ftest'][1]
            if p < 0.05:
                logging.info(f'Significant relation between {item} and PC2 ' f'(p={p:.4f}, lag={k}, timeframe={timeframe})')
            else:
                logging.info(f'H0 not rejected for {item}, p={p:.4f}, lag={k}, tf={timeframe}')
    return gc_list

def main():
    """
    Main runner:
    - Define timeframes and lags to test.
    - Execute Granger tests.
    """
    timeframes = ['uur', '4uur', 'dag']
    lags = [10]
    for i in timeframes:
        for j in lags:
            x = grangers(i, j)

if __name__ == '__main__':
    main()
