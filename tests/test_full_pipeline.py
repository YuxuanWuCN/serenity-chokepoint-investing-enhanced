import unittest
import numpy as np
import pandas as pd

from src.phase1.leading_indicator_tracker import FOITagger
from src.phase1.supply_chain_cross_validator import SupplyChainCrossValidator
from src.phase2.fama_macbeth import FamaMacBethRegressor, RegressionResult
from src.phase2.alpha_gate import AlphaGate, AlphaDecision
from src.phase3.zigzag_wave_analyzer import ZigZagWaveAnalyzer
from src.phase3.fibonacci_retracement_gate import FibonacciRetracementGate
from src.phase4.paper_portfolio_feedback import PaperPortfolioManager
from src.phase4.calibration_rlsp_loop import CalibrationRLSPLoop
from src.utils.data_fetcher import DataFetcher

class TestStockDashboardFullPipeline(unittest.TestCase):
    def test_phase1_foi_tagger(self):
        sample_text = """
        [FACT:Wind] 2025Q1 revenue grew 45% YoY.
        [OPINION:Analyst_A] Stock may double in next 12 months.
        [TRIANGULATED_FACT] Both TSMC and ASE confirmed advanced packaging capacity shortage.
        [INFERENCE:SupplyChain] Bottleneck lies in substrate layer.
        """
        tagger = FOITagger()
        report = tagger.validate_document(sample_text)
        self.assertEqual(report["total_statements"], 4)
        self.assertEqual(report["fact_count"], 2)
        self.assertTrue(report["has_triangulation"])
        self.assertTrue(report["is_grounded"])

    def test_phase1_cross_validator(self):
        validator = SupplyChainCrossValidator()
        target = {"ticker": "001258", "revenue_growth_yoy": 0.45, "prepayment_growth_yoy": 0.30}
        downstream = {"ticker": "CUSTOMER_A", "capex_growth_yoy": 0.42}
        upstream = {"ticker": "SUPPLIER_B", "contract_liabilities_growth_yoy": 0.28}

        signals = validator.validate_node_financials(target, downstream, upstream)
        self.assertEqual(len(signals), 2)
        self.assertTrue(all(s.signal_type == "RESONANCE" for s in signals))

    def test_phase2_fama_macbeth_and_gate(self):
        factors = DataFetcher.generate_synthetic_factors(n_periods=250, seed=42)
        stock = DataFetcher.generate_synthetic_stock(factors, true_alpha=0.0005, betas=(1.0, 0.2, 0.3, 0.1), residual_std=0.001, seed=42)
        reg = FamaMacBethRegressor(use_hac=True)
        res = reg.run_time_series_ols(stock, factors)
        
        gate = AlphaGate()
        eval_res = gate.evaluate(res)
        self.assertTrue(eval_res["passed"])
        self.assertEqual(eval_res["decision"], AlphaDecision.HIGH_CONVICTION)

    def test_phase3_zigzag_and_fibonacci(self):
        # Synthetic uptrend wave then wave 4 retrace
        prices = [10.0, 15.0, 12.0, 25.0, 17.5, 28.0] # Peak=25, Trough=12 -> Height=13, 0.5 retrace = 18.5, 0.618 = 16.96
        df_kline = pd.DataFrame({
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices
        }, index=pd.date_range("2025-01-01", periods=6, freq="D"))

        analyzer = ZigZagWaveAnalyzer(deviation_pct=0.15)
        points = analyzer.extract_zigzag_points(df_kline)
        self.assertTrue(len(points) >= 2)

        fibo_gate = FibonacciRetracementGate()
        decision = fibo_gate.evaluate_wave4_retrace(
            wave3_peak=25.0,
            wave2_trough=12.0,
            current_price=17.5, # Retrace = (25-17.5)/13 = 57.7% in [50%, 61.8%]
            current_atr=0.8
        )
        self.assertTrue(decision.passed)
        self.assertEqual(decision.signal_name, "WAVE4_FIBONACCI_CONFIRMED")

    def test_phase4_paper_portfolio_and_brier(self):
        pm = PaperPortfolioManager(initial_capital_cny=1_000_000.0)
        success = pm.open_position("Robust", "001258", price=100.0, allocation_pct=0.10)
        self.assertTrue(success)
        self.assertEqual(pm.robust_portfolio.positions["001258"].shares, 1000)

        # Price doubles to 200 -> triggers sell half
        events = pm.update_prices_and_rebalance({"001258": 200.0})
        self.assertEqual(len(events), 1)
        self.assertEqual(pm.robust_portfolio.positions["001258"].shares, 500)

        # Calibration loop check
        loop = CalibrationRLSPLoop()
        probs = [0.8, 0.7, 0.9, 0.2]
        actuals = [1, 1, 1, 0]
        metrics = loop.evaluate_predictions(probs, actuals)
        self.assertTrue(metrics.brier_score < 0.10)

    def test_phase1_financial_statements_df(self):
        validator = SupplyChainCrossValidator()
        target_df = pd.DataFrame({"ticker": ["001258"], "revenue_growth_yoy": [0.45], "prepayment_growth_yoy": [0.30]})
        downstream_df = pd.DataFrame({"ticker": ["TSMC"], "capex_growth_yoy": [0.40], "revenue_growth_yoy": [0.35]})
        upstream_df = pd.DataFrame({"ticker": ["SUPP"], "contract_liabilities_growth_yoy": [0.28]})
        signals = validator.validate_financial_statements_df(target_df, downstream_df, upstream_df)
        self.assertEqual(len(signals), 2)
        self.assertTrue(all(s.signal_type == "RESONANCE" for s in signals))

    def test_phase1_prompt_and_credibility(self):
        prompt = FOITagger.extract_foi_prompt_template("测试研报文本")
        self.assertIn("待分析文本", prompt)
        score_official = FOITagger.calculate_credibility_score("公司年报", "FACT")
        score_opinion = FOITagger.calculate_credibility_score("散户大V", "OPINION")
        self.assertTrue(score_official > score_opinion)

    def test_phase2_factor_providers_sqlite_cache(self):
        from src.utils.factor_providers import AkshareProxyFactorProvider, SyntheticFactorProvider
        prov = AkshareProxyFactorProvider(db_path="data/cache/test_factors.db")
        df = prov.get_factors(start_date="2024-01-01", end_date="2024-06-30")
        self.assertFalse(df.empty)
        self.assertIn("MKT_RF", df.columns)
        self.assertIn("SMB", df.columns)

    def test_phase3_causal_zigzag(self):
        prices = [10.0, 15.0, 12.0, 25.0, 17.5, 28.0]
        df_kline = pd.DataFrame({
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices
        }, index=pd.date_range("2025-01-01", periods=6, freq="D"))
        analyzer = ZigZagWaveAnalyzer(deviation_pct=0.10)
        causal_pivots = analyzer.confirm_pivots_causal(df_kline)
        self.assertTrue(len(causal_pivots) >= 2)

    def test_phase4_market_temperature_and_benchmark_matrix(self):
        from src.phase4.market_temperature import MarketTemperatureCalculator
        calc = MarketTemperatureCalculator()
        bm_ret = pd.Series(np.random.normal(0.001, 0.01, 50), index=pd.date_range("2025-01-01", periods=50, freq="B"))
        temp_res = calc.compute_temperature(bm_ret)
        self.assertIn("temperature", temp_res)
        self.assertTrue(0.0 <= temp_res["temperature"] <= 100.0)

        # Benchmark Matrix
        port_ret = bm_ret + 0.0005 # with positive alpha
        pool_df = pd.DataFrame({
            "S1": bm_ret + np.random.normal(0, 0.005, 50),
            "S2": bm_ret + np.random.normal(0, 0.005, 50),
            "S3": bm_ret + np.random.normal(0, 0.005, 50)
        })
        matrix_eval = PaperPortfolioManager.evaluate_with_benchmark_matrix(port_ret, pool_df, bm_ret, n_simulations=50)
        self.assertIn("excess_vs_broad_market", matrix_eval)
        self.assertTrue(matrix_eval["excess_vs_broad_market"] > 0)

    def test_end_to_end_serenity_pipeline(self):
        from src.pipeline import SerenityPipelineRunner
        runner = SerenityPipelineRunner()
        factors = DataFetcher.generate_synthetic_factors(n_periods=250, seed=42)
        stock = DataFetcher.generate_synthetic_stock(factors, true_alpha=0.001, betas=(1.0, 0.2, 0.1, 0.1), residual_std=0.001, seed=42)
        
        prices = np.cumprod(1 + stock) * 100.0
        kline = pd.DataFrame({
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices
        }, index=stock.index)

        sample_text = "[FACT:Wind] AI Semiconductor order surge.\n[TRIANGULATED_FACT] High conviction bottleneck."
        target_fin = {"ticker": "001258", "revenue_growth_yoy": 0.45, "prepayment_growth_yoy": 0.30}
        downstream_fin = {"ticker": "BIG_TECH", "capex_growth_yoy": 0.42}

        report = runner.run_full_pipeline(
            ticker="001258",
            unstructured_research_text=sample_text,
            stock_returns=stock,
            kline_df=kline,
            target_financials=target_fin,
            downstream_financials=downstream_fin
        )
        self.assertIn("stages", report)
        self.assertIn("phase1", report["stages"])
        self.assertIn("phase2", report["stages"])
        self.assertIn("phase3", report["stages"])
        self.assertIn("phase4", report["stages"])

    def test_eastmoney_miaoxiang_skill_and_provider(self):
        from src.skills.eastmoney_miaoxiang_skill import EastMoneyMiaoXiangSkill
        from src.utils.factor_providers import EastMoneyMiaoXiangProvider

        skill = EastMoneyMiaoXiangSkill(cache_db="data/cache/test_eastmoney_miaoxiang.db")
        
        # 1. 产业链图谱
        node = skill.get_supply_chain_ontology("001309")
        self.assertIsNotNone(node)
        self.assertEqual(node.target_name, "德明利")
        self.assertTrue(len(node.upstream_suppliers) > 0)

        # 2. 市场情绪快照
        breadth = skill.get_market_breadth_snapshot("2025-01-10")
        self.assertTrue(0.0 <= breadth.temperature_score <= 100.0)
        self.assertTrue(breadth.total_turnover_cny > 0)

        # 3. 因子适配器与微观资金流
        provider = EastMoneyMiaoXiangProvider(cache_db="data/cache/test_eastmoney_miaoxiang.db")
        df_factors = provider.get_factors_with_capital_flows("2025-01-01", "2025-01-10")
        self.assertFalse(df_factors.empty)
        self.assertIn("LARGE_ORDER_INFLOW", df_factors.columns)
        self.assertIn("NORTHBOUND_DELTA", df_factors.columns)
        self.assertIn("MKT_RF", df_factors.columns)

if __name__ == "__main__":
    unittest.main()
