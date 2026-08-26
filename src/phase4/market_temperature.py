import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

class MarketTemperatureCalculator:
    """
    Layer 4 Market Temperature (0 - 100°C) Index.
    Integrates Market Momentum, Breadth (advance/decline ratio), and Realized Volatility.
    
    Regimes:
    - 0°C - 30°C: Extreme Cold / Liquidity Freeze (Defensive, Max Equity <= 20%)
    - 30°C - 50°C: Cold / Accumulation (Max Equity <= 40%)
    - 50°C - 70°C: Neutral / Normal (Max Equity <= 70%)
    - 70°C - 85°C: Warm / Expansion (Max Equity <= 90%)
    - 85°C - 100°C: Overheated (Risk-off, Profit Taking, Max Equity <= 50%)
    """
    def __init__(self, momentum_window: int = 20, vol_window: int = 20):
        self.momentum_window = momentum_window
        self.vol_window = vol_window

    def compute_temperature(
        self,
        benchmark_returns: pd.Series,
        advance_decline_ratio: float = 0.5,
        analysis_date: Optional[str] = None
    ) -> Dict[str, Any]:
        if analysis_date is not None:
            benchmark_returns = benchmark_returns.loc[:pd.to_datetime(analysis_date)]

        returns = benchmark_returns.dropna()
        if len(returns) < self.momentum_window:
            # Default neutral temperature
            return {
                "temperature": 50.0,
                "regime": "NEUTRAL",
                "max_equity_allocation": 0.70,
                "momentum_score": 50.0,
                "volatility_penalty": 0.0,
                "description": "Insufficient history, default to Neutral 50.0°C"
            }

        # 1. Momentum Score (0 - 50)
        recent_ret = float(returns.tail(self.momentum_window).sum())
        norm_mom = np.clip((recent_ret + 0.10) / 0.20, 0.0, 1.0)
        mom_score = norm_mom * 50.0

        # 2. Breadth Score (0 - 30)
        breadth_score = np.clip(advance_decline_ratio, 0.0, 1.0) * 30.0

        # 3. Volatility Score (0 - 20)
        recent_vol = float(returns.tail(self.vol_window).std() * np.sqrt(252))
        norm_vol = np.clip(recent_vol / 0.40, 0.0, 1.0)
        vol_score = (1.0 - norm_vol) * 20.0

        total_temp = round(float(mom_score + breadth_score + vol_score), 1)
        total_temp = max(0.0, min(100.0, total_temp))

        if total_temp < 30.0:
            regime = "EXTREME_COLD"
            max_equity = 0.20
        elif total_temp < 50.0:
            regime = "COLD"
            max_equity = 0.40
        elif total_temp < 70.0:
            regime = "NEUTRAL"
            max_equity = 0.70
        elif total_temp < 85.0:
            regime = "WARM"
            max_equity = 0.90
        else:
            regime = "OVERHEATED"
            max_equity = 0.50

        return {
            "temperature": total_temp,
            "regime": regime,
            "max_equity_allocation": max_equity,
            "momentum_score": round(mom_score, 1),
            "breadth_score": round(breadth_score, 1),
            "volatility_score": round(vol_score, 1),
            "description": f"Market Temperature is {total_temp}°C ({regime}), recommended max equity {max_equity:.0%}."
        }
