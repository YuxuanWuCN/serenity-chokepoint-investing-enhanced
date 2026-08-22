import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

class DataFetcher:
    @staticmethod
    def generate_synthetic_factors(n_periods: int = 500, start_date: str = '2024-01-01', seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        dates = pd.date_range(start=start_date, periods=n_periods, freq='B')
        mkt_rf = np.random.normal(0.0004, 0.012, n_periods)
        smb = np.random.normal(0.0001, 0.007, n_periods)
        hml = np.random.normal(0.0001, 0.007, n_periods)
        mom = np.random.normal(0.0002, 0.008, n_periods)
        rf = np.full(n_periods, 0.00008)
        return pd.DataFrame({'MKT_RF': mkt_rf, 'SMB': smb, 'HML': hml, 'MOM': mom, 'RF': rf}, index=dates)

    @staticmethod
    def generate_synthetic_stock(factors_df: pd.DataFrame, true_alpha: float = 0.0002, betas: Tuple[float, float, float, float] = (1.1, 0.4, -0.3, 0.5), residual_std: float = 0.015, seed: int = 1258) -> pd.Series:
        np.random.seed(seed)
        n = len(factors_df)
        b_mkt, b_smb, b_hml, b_mom = betas
        eps = np.random.normal(0, residual_std, n)
        r_excess = (true_alpha + b_mkt * factors_df['MKT_RF'] + b_smb * factors_df['SMB'] + b_hml * factors_df['HML'] + b_mom * factors_df['MOM'] + eps)
        r_stock = r_excess + factors_df['RF']
        r_stock.name = 'return'
        return r_stock
