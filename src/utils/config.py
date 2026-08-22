"""Global configuration for StockDashboard v3.0 Quantitative Engine."""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AlphaGateConfig:
 p_value_threshold: float = 0.05
 ir_reject_threshold: float = 0.30
 ir_interesting_threshold: float = 0.50
 ir_attractive_threshold: float = 1.00

@dataclass
class KHunterConfig:
 fibo_retracements: tuple = (0.500, 0.618)
 atr_min_pct: float = 0.032 # 3.2%
 atr_max_pct: float = 0.095 # 9.5%
 mom_half_life: int = 20

@dataclass
class PaperPortfolioConfig:
 initial_capital_cny: float = 100.0
 robust_max_drawdown: float = 0.05
 aggressive_max_drawdown: float = 0.15
 brier_score_threshold: float = 0.25

DEFAULT_ALPHA_CONFIG = AlphaGateConfig()
DEFAULT_KHUNTER_CONFIG = KHunterConfig()
DEFAULT_PAPER_CONFIG = PaperPortfolioConfig()
