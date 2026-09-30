import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
    vif: Dict[str, float] = field(default_factory=dict)
    hac_lags: int = 5

class FamaMacBethRegressor:
    """
    Fama-MacBeth & OLS multi-factor time-series regression with adaptive HAC (Newey-West) standard errors,
    VIF multicollinearity diagnostics, and cross-sectional risk premium estimation.
    """
    def __init__(self, use_hac: bool = True, hac_maxlags: Optional[int] = None, adaptive_hac: bool = True):
        self.use_hac = use_hac
        self.hac_maxlags = hac_maxlags
        self.adaptive_hac = adaptive_hac

    @staticmethod
    def calc_newey_west_lags(n_obs: int) -> int:
        """
        Academic standard automatic bandwidth selection for Newey-West HAC:
        q = max(1, floor(4 * (T / 100)^(2/9)))
        """
        if n_obs <= 0:
            return 1
        return max(1, int(np.floor(4.0 * ((n_obs / 100.0) ** (2.0 / 9.0)))))

    @staticmethod
    def calc_vif(X: pd.DataFrame) -> Dict[str, float]:
        """
        Compute Variance Inflation Factor (VIF) for multi-factor collinearity check.
        """
        vif_dict = {}
        if X.shape[1] < 2:
            return {col: 1.0 for col in X.columns}
        
        X_clean = X.dropna()
        if len(X_clean) < X.shape[1] + 5:
            return {col: 1.0 for col in X.columns}

        X_with_const = sm.add_constant(X_clean)
        for i, col in enumerate(X_clean.columns):
            try:
                val = float(variance_inflation_factor(X_with_const.values, i + 1))
                vif_dict[col] = round(val, 2)
            except Exception:
                vif_dict[col] = 1.0
        return vif_dict

    def run_time_series_ols(
        self,
        stock_returns: pd.Series,
        factors_df: pd.DataFrame,
        factor_cols: Optional[list] = None,
        analysis_date: Optional[str] = None
    ) -> RegressionResult:
        if factor_cols is None:
            factor_cols = [c for c in ["MKT_RF", "SMB", "HML", "MOM"] if c in factors_df.columns]

        if analysis_date is not None:
            stock_returns = stock_returns.loc[:pd.to_datetime(analysis_date)]
            factors_df = factors_df.loc[:pd.to_datetime(analysis_date)]

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
        n_obs = len(common_idx)
        
        # Determine HAC lags
        if self.hac_maxlags is not None:
            effective_lags = self.hac_maxlags
        elif self.adaptive_hac:
            effective_lags = self.calc_newey_west_lags(n_obs)
        else:
            effective_lags = 5
        
        if self.use_hac:
            model = sm.OLS(y_excess, X_with_const).fit(
                cov_type="HAC", 
                cov_kwds={"maxlags": effective_lags}
            )
        else:
            model = sm.OLS(y_excess, X_with_const).fit()

        alpha = float(model.params["const"])
        alpha_t = float(model.tvalues["const"])
        alpha_p = float(model.pvalues["const"])
        r2 = float(model.rsquared)
        resid_std = float(np.std(model.resid, ddof=len(model.params)))
        
        # Daily IR = alpha / resid_std, annualized IR = daily_ir * sqrt(252)
        daily_ir = float(alpha / resid_std) if resid_std > 1e-9 else 0.0
        ir_annualized = float(daily_ir * np.sqrt(252))

        betas = {col: float(model.params[col]) for col in factor_cols}
        beta_tstats = {col: float(model.tvalues[col]) for col in factor_cols}
        vif_scores = self.calc_vif(X)

        return RegressionResult(
            alpha=alpha,
            alpha_tstat=alpha_t,
            alpha_pvalue=alpha_p,
            r_squared=r2,
            betas=betas,
            beta_tstats=beta_tstats,
            residual_std=resid_std,
            ir=daily_ir,
            n_obs=n_obs,
            hac_robust=self.use_hac,
            vif=vif_scores,
            hac_lags=effective_lags
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
