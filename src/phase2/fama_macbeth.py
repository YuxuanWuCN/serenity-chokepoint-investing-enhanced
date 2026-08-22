import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class RegressionResult:
    alpha: float
    alpha_tstat: float
    alpha_pvalue: float
    r_squared: float
    betas: Dict[str, float]
    beta_tstats: Dict[str, float]
    residual_std: float
    ir: float
    n_obs: int
    hac_robust: bool

class FamaMacBethRegressor:
    """
    Fama-MacBeth & OLS multi-factor time-series regression with HAC (Newey-West) standard errors.
    """
    def __init__(self, use_hac: bool = True, hac_maxlags: int = 5):
        self.use_hac = use_hac
        self.hac_maxlags = hac_maxlags

    def run_time_series_ols(
        self,
        stock_returns: pd.Series,
        factors_df: pd.DataFrame,
        factor_cols: Optional[list] = None
    ) -> RegressionResult:
        if factor_cols is None:
            factor_cols = [c for c in ["MKT_RF", "SMB", "HML", "MOM"] if c in factors_df.columns]

        common_idx = stock_returns.dropna().index.intersection(factors_df.dropna().index)
        if len(common_idx) < 30:
            raise ValueError(f"Insufficient common data points: {len(common_idx)} < 30")

        y = stock_returns.loc[common_idx]
        X = factors_df.loc[common_idx, factor_cols]
        
        # If stock_returns is total return and RF is provided, compute excess return y_excess = y - RF
        if "RF" in factors_df.columns:
            y_excess = y - factors_df.loc[common_idx, "RF"]
        else:
            y_excess = y

        X_with_const = sm.add_constant(X)
        
        if self.use_hac:
            model = sm.OLS(y_excess, X_with_const).fit(
                cov_type="HAC", 
                cov_kwds={"maxlags": self.hac_maxlags}
            )
        else:
            model = sm.OLS(y_excess, X_with_const).fit()

        alpha = float(model.params["const"])
        alpha_t = float(model.tvalues["const"])
        alpha_p = float(model.pvalues["const"])
        r2 = float(model.rsquared)
        resid_std = float(np.std(model.resid, ddof=len(model.params)))
        
        # Daily IR = alpha / resid_std, annualized IR = (alpha * 252) / (resid_std * np.sqrt(252)) = IR_daily * sqrt(252)
        # In financial practice or daily terms, calculate annualized IR:
        ir_annualized = float((alpha / resid_std) * np.sqrt(252)) if resid_std > 1e-9 else 10.0

        betas = {col: float(model.params[col]) for col in factor_cols}
        beta_tstats = {col: float(model.tvalues[col]) for col in factor_cols}

        return RegressionResult(
            alpha=alpha,
            alpha_tstat=alpha_t,
            alpha_pvalue=alpha_p,
            r_squared=r2,
            betas=betas,
            beta_tstats=beta_tstats,
            residual_std=resid_std,
            ir=ir_annualized,
            n_obs=len(common_idx),
            hac_robust=self.use_hac
        )

    def run_gmm_robustness(
        self,
        stock_returns: pd.Series,
        factors_df: pd.DataFrame,
        split_ratio: float = 0.5
    ) -> Dict[str, Any]:
        common_idx = stock_returns.dropna().index.intersection(factors_df.dropna().index)
        n = len(common_idx)
        split_pt = int(n * split_ratio)
        
        idx1 = common_idx[:split_pt]
        idx2 = common_idx[split_pt:]
        
        res1 = self.run_time_series_ols(stock_returns.loc[idx1], factors_df.loc[idx1])
        res2 = self.run_time_series_ols(stock_returns.loc[idx2], factors_df.loc[idx2])
        
        persistent = (res1.alpha * res2.alpha > 0) and (res1.alpha_pvalue < 0.10 and res2.alpha_pvalue < 0.10)
        stability_gap = abs(res1.alpha - res2.alpha)
        
        return {
            "period1":{"alpha": res1.alpha, "pvalue": res1.alpha_pvalue, "ir": res1.ir},
            "period2":{"alpha": res2.alpha, "pvalue": res2.alpha_pvalue, "ir": res2.ir},
            "is_persistent": persistent,
            "alpha_gap": stability_gap
        }
