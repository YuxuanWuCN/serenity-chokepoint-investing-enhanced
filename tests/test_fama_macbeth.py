import unittest
import numpy as np
import pandas as pd
from src.utils.data_fetcher import DataFetcher
from src.phase2.fama_macbeth import FamaMacBethRegressor, RegressionResult
from src.phase2.alpha_gate import AlphaGate, AlphaDecision

class TestFamaMacBeth(unittest.TestCase):
    def test_fama_macbeth_regression(self):
        factors = DataFetcher.generate_synthetic_factors(n_periods=250, seed=42)
        stock = DataFetcher.generate_synthetic_stock(factors, true_alpha=0.0005, betas=(1.0, 0.2, 0.3, 0.1), residual_std=0.001, seed=42)
        reg = FamaMacBethRegressor(use_hac=True)
        res = reg.run_time_series_ols(stock, factors)
        self.assertIsInstance(res, RegressionResult)
        self.assertEqual(res.n_obs, 250)
        self.assertTrue(abs(res.betas["MKT_RF"] - 1.0) < 0.15)

    def test_alpha_gate_reject_001258_case(self):
        dummy_res = RegressionResult(
            alpha=0.0017,
            alpha_tstat=0.93,
            alpha_pvalue=0.3543,
            r_squared=0.12,
            betas={"MKT_RF": 1.0, "HML": -0.391},
            beta_tstats={"MKT_RF": 3.5, "HML": -1.2},
            residual_std=0.027,
            ir=0.063,
            n_obs=250,
            hac_robust=True
        )
        gate = AlphaGate()
        eval_res = gate.evaluate(dummy_res)
        self.assertFalse(eval_res["passed"])
        self.assertEqual(eval_res["decision"], AlphaDecision.REJECT_NON_SIGNIFICANT)

    def test_alpha_gate_high_conviction(self):
        dummy_res = RegressionResult(
            alpha=0.002,
            alpha_tstat=3.5,
            alpha_pvalue=0.0005,
            r_squared=0.45,
            betas={"MKT_RF": 1.1, "HML": 0.1},
            beta_tstats={"MKT_RF": 5.0, "HML": 1.0},
            residual_std=0.0015,
            ir=1.333,
            n_obs=250,
            hac_robust=True
        )
        gate = AlphaGate()
        eval_res = gate.evaluate(dummy_res)
        self.assertTrue(eval_res["passed"])
        self.assertEqual(eval_res["decision"], AlphaDecision.HIGH_CONVICTION)

    def test_adaptive_hac_lags(self):
        # Formula: max(1, floor(4 * (T / 100)^(2/9)))
        self.assertEqual(FamaMacBethRegressor.calc_newey_west_lags(100), 4)
        self.assertEqual(FamaMacBethRegressor.calc_newey_west_lags(250), 4)
        self.assertEqual(FamaMacBethRegressor.calc_newey_west_lags(500), 5)
        self.assertEqual(FamaMacBethRegressor.calc_newey_west_lags(10), 2)

    def test_vif_calculation(self):
        factors = DataFetcher.generate_synthetic_factors(n_periods=250, seed=42)
        vif_dict = FamaMacBethRegressor.calc_vif(factors[["MKT_RF", "SMB", "HML", "MOM"]])
        self.assertIn("MKT_RF", vif_dict)
        self.assertIn("SMB", vif_dict)
        for val in vif_dict.values():
            self.assertTrue(val < 5.0) # Synthetic factors are orthogonal, VIF ~ 1.0

if __name__ == '__main__':
    unittest.main()