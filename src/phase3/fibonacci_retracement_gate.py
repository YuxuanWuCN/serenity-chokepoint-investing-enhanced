import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from src.phase3.zigzag_wave_analyzer import ZigZagPoint
from src.utils.config import KHunterConfig, DEFAULT_KHUNTER_CONFIG

@dataclass
class FibonacciGateDecision:
    passed: bool
    signal_name: str
    retrace_ratio: float
    target_entry_low: float
    target_entry_high: float
    current_price: float
    stop_loss: float
    take_profit_double_reduction: float
    description: str

class FibonacciRetracementGate:
    """
    Layer 3 Fibonacci 0.500 / 0.618 Retracement Gate + ATR-based Stop Loss.
    """
    def __init__(self, config: Optional[KHunterConfig] = None):
        self.config = config or DEFAULT_KHUNTER_CONFIG

    def evaluate_wave4_retrace(
        self,
        wave3_peak: float,
        wave2_trough: float,
        current_price: float,
        current_atr: float,
        atr_multiplier: float = 2.5
    ) -> FibonacciGateDecision:
        wave_height = wave3_peak - wave2_trough
        if wave_height <= 0:
            return FibonacciGateDecision(
                passed=False,
                signal_name="INVALID_WAVE",
                retrace_ratio=0.0,
                target_entry_low=0.0,
                target_entry_high=0.0,
                current_price=current_price,
                stop_loss=0.0,
                take_profit_double_reduction=0.0,
                description="Invalid wave height: peak must be strictly greater than trough."
            )

        fibo_500 = wave3_peak - 0.500 * wave_height
        fibo_618 = wave3_peak - 0.618 * wave_height

        target_low = min(fibo_500, fibo_618)
        target_high = max(fibo_500, fibo_618)

        # Retracement ratio of current price from peak
        retrace_ratio = (wave3_peak - current_price) / wave_height

        # Check if price is within [0.500, 0.618] buffer (+/- 5%)
        in_retrace_zone = (0.45 <= retrace_ratio <= 0.68)
        
        # Stop loss based on 2.5 * ATR or below wave 2 trough
        stop_loss = max(wave2_trough, current_price - atr_multiplier * current_atr)
        
        # Double reduction price (100% gain lock)
        take_profit_double = current_price * 2.0

        if in_retrace_zone:
            return FibonacciGateDecision(
                passed=True,
                signal_name="WAVE4_FIBONACCI_CONFIRMED",
                retrace_ratio=retrace_ratio,
                target_entry_low=target_low,
                target_entry_high=target_high,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit_double_reduction=take_profit_double,
                description=f"Price {current_price:.2f} is in optimal Fibonacci 0.500-0.618 retracement zone ({target_low:.2f} - {target_high:.2f})."
            )
        else:
            return FibonacciGateDecision(
                passed=False,
                signal_name="RETRACE_OUT_OF_BOUNDS",
                retrace_ratio=retrace_ratio,
                target_entry_low=target_low,
                target_entry_high=target_high,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit_double_reduction=take_profit_double,
                description=f"Price {current_price:.2f} (retrace {retrace_ratio:.1%}) is outside Fibonacci buy zone [50.0%, 61.8%]."
            )
