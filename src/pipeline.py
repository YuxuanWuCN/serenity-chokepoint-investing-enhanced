import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.phase1.leading_indicator_tracker import FOITagger
from src.phase1.supply_chain_cross_validator import SupplyChainCrossValidator
from src.phase2.fama_macbeth import FamaMacBethRegressor, RegressionResult
from src.phase2.alpha_gate import AlphaGate, AlphaDecision
from src.phase3.zigzag_wave_analyzer import ZigZagWaveAnalyzer
from src.phase3.fibonacci_retracement_gate import FibonacciRetracementGate
from src.phase4.market_temperature import MarketTemperatureCalculator
from src.phase4.paper_portfolio_feedback import PaperPortfolioManager
from src.utils.factor_providers import BaseFactorProvider, SyntheticFactorProvider

class SerenityPipelineRunner:
    """
    End-to-End Two-Stage Quantitative Investment Pipeline Orchestrator.
    Executes:
    1. Phase 1: FOI Qualitative Analysis & Supply Chain Cross-Validation
    2. Phase 2: Factor Time-Series Regression & Alpha Gate Interception
    3. Phase 3: KHunter Wave 4 & Fibonacci Retracement Timing Gate
    4. Phase 4: Market Temperature Gating & Paper Portfolio Execution
    """
    def __init__(
        self,
        factor_provider: Optional[BaseFactorProvider] = None,
        initial_capital_cny: float = 1_000_000.0
    ):
        self.foi_tagger = FOITagger()
        self.cross_validator = SupplyChainCrossValidator()
        self.regressor = FamaMacBethRegressor(use_hac=True, adaptive_hac=True)
        self.alpha_gate = AlphaGate()
        self.wave_analyzer = ZigZagWaveAnalyzer(deviation_pct=0.10)
        self.fibo_gate = FibonacciRetracementGate()
        self.temp_calculator = MarketTemperatureCalculator()
        self.portfolio_manager = PaperPortfolioManager(initial_capital_cny=initial_capital_cny)
        self.factor_provider = factor_provider or SyntheticFactorProvider(seed=42)

    def run_full_pipeline(
        self,
        ticker: str,
        unstructured_research_text: str,
        stock_returns: pd.Series,
        kline_df: pd.DataFrame,
        target_financials: Dict[str, float],
        downstream_financials: Dict[str, float],
        upstream_financials: Optional[Dict[str, float]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        bet_type: str = "Catalyst Alpha"
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "ticker": ticker,
            "bet_type": bet_type,
            "overall_decision": "REJECT",
            "stages": {}
        }

        # Stage 1: Qualitative FOI Validation
        foi_report = self.foi_tagger.validate_document(unstructured_research_text)
        chain_signals = self.cross_validator.validate_node_financials(
            target_financials, downstream_financials, upstream_financials
        )
        report["stages"]["phase1"] = {
            "foi": foi_report,
            "supply_chain_signals": [s.__dict__ for s in chain_signals],
            "passed": foi_report.get("is_grounded", False)
        }

        # Stage 2: Factor Regression & Alpha Gate
        factors_df = self.factor_provider.get_factors(
            start_date=str(stock_returns.index.min())[:10],
            end_date=str(stock_returns.index.max())[:10]
        )
        reg_result = self.regressor.run_time_series_ols(stock_returns, factors_df)
        alpha_eval = self.alpha_gate.evaluate(reg_result)
        report["stages"]["phase2"] = {
            "alpha": reg_result.alpha,
            "alpha_pvalue": reg_result.alpha_pvalue,
            "ir": reg_result.ir,
            "hac_lags": reg_result.hac_lags,
            "vif": reg_result.vif,
            "gate_decision": alpha_eval["decision"].value,
            "passed": alpha_eval["passed"]
        }

        if not alpha_eval["passed"]:
            report["reason"] = f"Failed Phase 2 Alpha Gate: {alpha_eval['description']}"
            return report

        # Stage 3: KHunter Wave 4 Timing Gate
        pivots = self.wave_analyzer.confirm_pivots_causal(kline_df)
        current_p = float(kline_df["close"].iloc[-1])
        current_atr = float(self.wave_analyzer.compute_atr(kline_df).iloc[-1]) if len(kline_df) >= 14 else 1.0

        # Find last Peak and preceding Trough
        peaks = [p for p in pivots if p.point_type == "PEAK"]
        troughs = [p for p in pivots if p.point_type == "TROUGH"]

        if peaks and troughs:
            w3_peak = peaks[-1].price
            w2_trough = troughs[-1].price
            fibo_eval = self.fibo_gate.evaluate_wave4_retrace(
                wave3_peak=w3_peak,
                wave2_trough=w2_trough,
                current_price=current_p,
                current_atr=current_atr
            )
        else:
            fibo_eval = self.fibo_gate.evaluate_wave4_retrace(
                wave3_peak=current_p * 1.2,
                wave2_trough=current_p * 0.8,
                current_price=current_p,
                current_atr=current_atr
            )

        report["stages"]["phase3"] = {
            "current_price": current_p,
            "atr": current_atr,
            "retrace_ratio": fibo_eval.retrace_ratio,
            "stop_loss": fibo_eval.stop_loss,
            "signal_name": fibo_eval.signal_name,
            "passed": fibo_eval.passed
        }

        # Stage 4: Market Temperature & Paper Allocation
        if benchmark_returns is None:
            benchmark_returns = factors_df["MKT_RF"]
        
        temp_eval = self.temp_calculator.compute_temperature(benchmark_returns)
        self.portfolio_manager.apply_market_temperature(temp_eval["temperature"])
        
        opened_robust = self.portfolio_manager.open_position(
            portfolio_type="Robust",
            ticker=ticker,
            price=current_p,
            allocation_pct=0.10,
            bet_type=bet_type
        )
        opened_aggressive = self.portfolio_manager.open_position(
            portfolio_type="Aggressive",
            ticker=ticker,
            price=current_p,
            allocation_pct=0.15,
            bet_type=bet_type
        )

        report["stages"]["phase4"] = {
            "market_temperature": temp_eval,
            "opened_robust": opened_robust,
            "opened_aggressive": opened_aggressive,
            "portfolio_equity": {
                "Robust": self.portfolio_manager.robust_portfolio.total_equity,
                "Aggressive": self.portfolio_manager.aggressive_portfolio.total_equity
            }
        }

        report["overall_decision"] = "APPROVED_FOR_ALLOCATION" if (alpha_eval["passed"] and fibo_eval.passed) else "HOLD_WAIT_FOR_RETRACE"
        return report
