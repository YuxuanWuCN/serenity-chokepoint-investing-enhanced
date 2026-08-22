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

    @property
    def total_equity(self) -> float:
        pos_val = sum(p.shares * p.current_price for p in self.positions.values())
        return self.cash_cny + pos_val

class PaperPortfolioManager:
    """
    Layer 4 Paper Trading Portfolio Tracker (Robust vs Aggressive).
    Enforces initial capital (e.g. 1,000,000 CNY) and double-reduction rule.
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

    def open_position(
        self,
        portfolio_type: str,
        ticker: str,
        price: float,
        allocation_pct: float,
        bet_type: str = "Catalyst Alpha"
    ) -> bool:
        port = self.robust_portfolio if portfolio_type == "Robust" else self.aggressive_portfolio
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
