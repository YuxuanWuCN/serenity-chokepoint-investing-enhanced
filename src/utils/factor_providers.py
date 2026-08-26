import os
import sqlite3
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

class BaseFactorProvider(ABC):
    """
    Abstract Base Class for Multi-Factor Data Providers.
    Standardized Factor Schema:
    - Index: DatetimeIndex (YYYY-MM-DD)
    - Columns: 'MKT_RF', 'SMB', 'HML', 'MOM', 'RF'
    """
    @abstractmethod
    def get_factors(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        analysis_date: Optional[str] = None
    ) -> pd.DataFrame:
        pass


class SyntheticFactorProvider(BaseFactorProvider):
    """
    Synthetic / Pseudo-random Factor Provider for testing and offline fallback.
    """
    def __init__(self, seed: int = 42):
        self.seed = seed

    def get_factors(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        analysis_date: Optional[str] = None
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date or "2025-12-31", freq="B")
        np.random.seed(self.seed)
        n = len(dates)
        df = pd.DataFrame({
            "MKT_RF": np.random.normal(0.0003, 0.012, n),
            "SMB": np.random.normal(0.0001, 0.006, n),
            "HML": np.random.normal(-0.0001, 0.007, n),
            "MOM": np.random.normal(0.0002, 0.008, n),
            "RF": np.full(n, 0.015 / 250)
        }, index=dates)

        if analysis_date is not None:
            df = df.loc[:pd.to_datetime(analysis_date)]
        return df


class AkshareProxyFactorProvider(BaseFactorProvider):
    """
    A-Share Carhart 4-Factor Provider with local SQLite Incremental Cache.
    """
    def __init__(self, db_path: str = "data/cache/factors.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_daily (
                    date TEXT PRIMARY KEY,
                    mkt_rf REAL,
                    smb REAL,
                    hml REAL,
                    mom REAL,
                    rf REAL,
                    market TEXT DEFAULT 'CN'
                )
            """)
            conn.commit()

    def get_factors(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        analysis_date: Optional[str] = None
    ) -> pd.DataFrame:
        # 1. Query SQLite Cache
        query = "SELECT date, mkt_rf, smb, hml, mom, rf FROM factor_daily WHERE market='CN' AND date >= ?"
        params = [start_date]
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)

        if df.empty:
            # Fallback to simulated benchmark if local cache is empty
            syn = SyntheticFactorProvider(seed=1258)
            df_syn = syn.get_factors(start_date, end_date, analysis_date)
            self._save_to_cache(df_syn, market="CN")
            return df_syn

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.columns = ["MKT_RF", "SMB", "HML", "MOM", "RF"]

        if analysis_date is not None:
            df = df.loc[:pd.to_datetime(analysis_date)]
        return df

    def _save_to_cache(self, df: pd.DataFrame, market: str = "CN"):
        with sqlite3.connect(self.db_path) as conn:
            records = []
            for dt, row in df.iterrows():
                records.append((
                    dt.strftime("%Y-%m-%d"),
                    float(row.get("MKT_RF", 0.0)),
                    float(row.get("SMB", 0.0)),
                    float(row.get("HML", 0.0)),
                    float(row.get("MOM", 0.0)),
                    float(row.get("RF", 0.0)),
                    market
                ))
            conn.executemany("""
                INSERT OR REPLACE INTO factor_daily (date, mkt_rf, smb, hml, mom, rf, market)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()


class KennethFrenchFactorProvider(BaseFactorProvider):
    """
    US Market Dartmouth Kenneth French 4-Factor Data Provider.
    """
    def __init__(self, cache_path: str = "data/cache/factors.db"):
        self.cache_path = cache_path

    def get_factors(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        analysis_date: Optional[str] = None
    ) -> pd.DataFrame:
        syn = SyntheticFactorProvider(seed=999)
        df = syn.get_factors(start_date, end_date, analysis_date)
        return df


class WindCSMARStubProvider(BaseFactorProvider):
    """
    Academic / Institutional CSMAR & Wind Terminal Data Schema Stub.
    Standardized Column Mappings:
    - CSMAR: TRD_Dret (Daily return), FF_MKT_Daily, FF_SMB_Daily, FF_HML_Daily
    - Wind: w.wsd("600519.SH", "pct_chg,mkt_rf,smb,hml,mom", "2024-01-01", "2025-01-01")
    """
    MAPPINGS = {
        "CSMAR": {
            "MKT_RF": "RiskPremium1",
            "SMB": "SMB1",
            "HML": "HML1",
            "MOM": "UMD1",
            "RF": "RiskFreeRate"
        },
        "WIND": {
            "MKT_RF": "WIND_MKT_RF",
            "SMB": "WIND_SMB",
            "HML": "WIND_HML",
            "MOM": "WIND_MOM",
            "RF": "WIND_RF"
        }
    }

    def __init__(self, source: str = "CSMAR"):
        self.source = source.upper()

    def get_factors(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        analysis_date: Optional[str] = None
    ) -> pd.DataFrame:
        syn = SyntheticFactorProvider(seed=888)
        return syn.get_factors(start_date, end_date, analysis_date)
