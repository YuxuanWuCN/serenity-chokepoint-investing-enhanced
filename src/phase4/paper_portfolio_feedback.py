import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class Position:
    ticker: str
    shares: int
    entry_price: float
    current_price: float
    bet_type: str  # 'Super Beta', 'Catalyst Alpha', 'Event-Driven'
    unrealized_pnl: float = 0.0
    highest_price: float = 0.0
    is_doubled_halved: bool = False

@dataclass
class PaperPortfolio:
    name: str  # 'Robust' or 'Aggressive'
    cash_cny: float
    initial_capital_cny: float
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    max_equity_pct: float = 1.0

    @property
    def total_equity(self) -> float:
        pos_val = sum(p.shares * p.current_price for p in self.positions.values())
        return self.cash_cny + pos_val

    @property
    def current_equity_exposure(self) -> float:
        tot = self.total_equity
        if tot <= 0:
            return 0.0
        pos_val = sum(p.shares * p.current_price for p in self.positions.values())
        return pos_val / tot

class PaperPortfolioManager:
    """
    Layer 4 Paper Trading Portfolio Tracker (Robust vs Aggressive).
    Enforces initial capital (e.g. 1,000,000 CNY), double-reduction rule,
    market-temperature dynamic cash gating, and 3-tier benchmark matrix evaluation.
    """
    def __init__(self, initial_capital_cny: float = 1_000_000.0):
        self.robust_portfolio = PaperPortfolio(
            name="Robust",
            cash_cny=initial_capital_cny,
            initial_capital_cny=initial_capital_cny
        )
        self.aggressive_portfolio = PaperPortfolio(
            name="Aggressive",
            cash_cny=initial_capital_cny,
            initial_capital_cny=initial_capital_cny
        )

    def apply_market_temperature(self, temperature: float) -> Dict[str, float]:
        """
        Dynamically adjusts max equity budget according to 0 - 100°C market temperature.
        """
        if temperature < 30.0:
            max_eq = 0.20
        elif temperature < 50.0:
            max_eq = 0.40
        elif temperature < 70.0:
            max_eq = 0.70
        elif temperature < 85.0:
            max_eq = 0.90
        else:
            max_eq = 0.50

        self.robust_portfolio.max_equity_pct = max_eq * 0.8  # Robust is more defensive
        self.aggressive_portfolio.max_equity_pct = max_eq
        return {
            "Robust_max_equity": self.robust_portfolio.max_equity_pct,
            "Aggressive_max_equity": self.aggressive_portfolio.max_equity_pct
        }

    def open_position(
        self,
        portfolio_type: str,
        ticker: str,
        price: float,
        allocation_pct: float,
        bet_type: str = "Catalyst Alpha"
    ) -> bool:
        port = self.robust_portfolio if portfolio_type == "Robust" else self.aggressive_portfolio
        
        # Check temperature-gated max equity constraint
        current_exposure = port.current_equity_exposure
        if current_exposure + allocation_pct > port.max_equity_pct:
            # Scale down allocation to fit within temperature risk budget
            allocation_pct = max(0.0, port.max_equity_pct - current_exposure)
            if allocation_pct <= 0.01:
                return False

        alloc_amount = port.total_equity * allocation_pct
        if alloc_amount > port.cash_cny or price <= 0:
            return False

        shares = int(alloc_amount / price)
        if shares <= 0:
            return False

        cost = shares * price
        port.cash_cny -= cost
        port.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            current_price=price,
            bet_type=bet_type,
            highest_price=price
        )
        return True

    def update_prices_and_rebalance(self, price_dict: Dict[str, float]) -> List[str]:
        events = []
        for port in [self.robust_portfolio, self.aggressive_portfolio]:
            for ticker, pos in list(port.positions.items()):
                if ticker in price_dict:
                    new_p = price_dict[ticker]
                    pos.current_price = new_p
                    pos.highest_price = max(pos.highest_price, new_p)
                    pos.unrealized_pnl = (new_p - pos.entry_price) * pos.shares

                    # 100% gain -> sell half to lock principal
                    if new_p >= pos.entry_price * 2.0 and not pos.is_doubled_halved:
                        half_shares = pos.shares // 2
                        proceeds = half_shares * new_p
                        pos.shares -= half_shares
                        port.cash_cny += proceeds
                        pos.is_doubled_halved = True
                        events.append(f"[{port.name}] {ticker} doubled! Sold {half_shares} shares, locked {proceeds:.2f} CNY.")

        return events

    def record_daily_nav(self, date: str, price_dict: Optional[Dict[str, float]] = None):
        if price_dict:
            self.update_prices_and_rebalance(price_dict)
        for port in [self.robust_portfolio, self.aggressive_portfolio]:
            port.history.append({
                "date": date,
                "total_equity": port.total_equity,
                "cash": port.cash_cny,
                "positions_count": len(port.positions),
                "nav": port.total_equity / port.initial_capital_cny
            })

    @staticmethod
    def evaluate_with_benchmark_matrix(
        portfolio_returns: pd.Series,
        pool_returns_df: pd.DataFrame,
        benchmark_returns: pd.Series,
        n_simulations: int = 100,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        3-Tier Scientific Benchmark Control Matrix:
        1. Broad Market Index Benchmark (e.g. CSI 300)
        2. Equal-Weight Candidate Pool Benchmark
        3. Monte Carlo 100-run Random Selection Control Group (p-value of Alpha)
        """
        common_idx = portfolio_returns.dropna().index.intersection(benchmark_returns.dropna().index)
        if len(common_idx) < 10:
            return {"status": "insufficient_data"}

        p_ret = portfolio_returns.loc[common_idx]
        b_ret = benchmark_returns.loc[common_idx]

        # 1. Broad Market Metrics
        cum_p = float((1 + p_ret).prod() - 1)
        cum_b = float((1 + b_ret).prod() - 1)
        excess_vs_bm = cum_p - cum_b

        # 2. Equal-Weight Pool Benchmark
        pool_common = pool_returns_df.loc[pool_returns_df.index.intersection(common_idx)]
        eq_pool_ret = pool_common.mean(axis=1) if not pool_common.empty else b_ret
        cum_eq_pool = float((1 + eq_pool_ret).prod() - 1)
        excess_vs_pool = cum_p - cum_eq_pool

        # 3. Monte Carlo Random Control Simulation
        np.random.seed(seed)
        sim_returns = []
        if not pool_common.empty and pool_common.shape[1] >= 3:
            cols = list(pool_common.columns)
            k = min(3, len(cols))
            for _ in range(n_simulations):
                sampled_cols = np.random.choice(cols, size=k, replace=False)
                sim_ret = pool_common[sampled_cols].mean(axis=1)
                sim_returns.append(float((1 + sim_ret).prod() - 1))
            
            beat_count = sum(1 for r in sim_returns if cum_p > r)
            p_val = 1.0 - (beat_count / n_simulations)
            sim_mean = float(np.mean(sim_returns))
        else:
            p_val = 0.05
            sim_mean = cum_b

        return {
            "portfolio_cum_return": round(cum_p, 4),
            "broad_market_cum_return": round(cum_b, 4),
            "excess_vs_broad_market": round(excess_vs_bm, 4),
            "equal_weight_pool_return": round(cum_eq_pool, 4),
            "excess_vs_equal_weight": round(excess_vs_pool, 4),
            "monte_carlo_control_mean": round(sim_mean, 4),
            "monte_carlo_p_value": round(p_val, 4),
            "is_alpha_statistically_significant": p_val < 0.05
        }

