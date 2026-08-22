import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ZigZagPoint:
    index: int
    date: Any
    price: float
    point_type: str  # 'PEAK' or 'TROUGH'

class ZigZagWaveAnalyzer:
    """
    Layer 3 KHunter ZigZag wave segmenter and Elliott Wave 4 identification.
    """
    def __init__(self, deviation_pct: float = 0.05):
        self.deviation_pct = deviation_pct

    def extract_zigzag_points(self, df: pd.DataFrame, price_col: str = "close") -> List[ZigZagPoint]:
        prices = df[price_col].values
        dates = df.index.values
        n = len(prices)
        if n < 5:
            return []

        points = []
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        trend = 0  # 1 for up, -1 for down, 0 initial

        for i in range(1, n):
            change = (prices[i] - last_pivot_price) / last_pivot_price
            if trend == 0:
                if change >= self.deviation_pct:
                    trend = 1
                    points.append(ZigZagPoint(0, dates[0], prices[0], 'TROUGH'))
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
                elif change <= -self.deviation_pct:
                    trend = -1
                    points.append(ZigZagPoint(0, dates[0], prices[0], 'PEAK'))
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
            elif trend == 1:
                if prices[i] > last_pivot_price:
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
                elif change <= -self.deviation_pct:
                    points.append(ZigZagPoint(last_pivot_idx, dates[last_pivot_idx], last_pivot_price, 'PEAK'))
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
            elif trend == -1:
                if prices[i] < last_pivot_price:
                    last_pivot_idx = i
                    last_pivot_price = prices[i]
                elif change >= self.deviation_pct:
                    points.append(ZigZagPoint(last_pivot_idx, dates[last_pivot_idx], last_pivot_price, 'TROUGH'))
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_price = prices[i]

        if points:
            final_type = 'PEAK' if trend == 1 else 'TROUGH'
            points.append(ZigZagPoint(last_pivot_idx, dates[last_pivot_idx], last_pivot_price, final_type))

        return points

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr
