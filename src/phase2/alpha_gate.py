import pandas as pd
from typing import Dict, Any, Optional
from enum import Enum
from src.phase2.fama_macbeth import RegressionResult
from src.utils.config import AlphaGateConfig, DEFAULT_ALPHA_CONFIG

class AlphaDecision(str, Enum):
    HIGH_CONVICTION = "HIGH_CONVICTION"
    ATTRACTIVE = "ATTRACTIVE"
    INTERESTING_SMALL = "INTERESTING_SMALL"
    REJECT_NON_SIGNIFICANT = "REJECT_NON_SIGNIFICANT"
    REJECT_LOW_IR = "REJECT_LOW_IR"
    REJECT_NON_PERSISTENT = "REJECT_NON_PERSISTENT"

class AlphaGate:
    """
    Alpha Gate filter
    """
    def __init__(self, config: Optional[AlphaGateConfig] = None):
        self.config = config or DEFAULT_ALPHA_CONFIG

    def evaluate(
        self,
        reg_result: RegressionResult,
        gmm_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if reg_result.alpha_pvalue > self.config.p_value_threshold:
            return {
                "passed": False,
                "decision": AlphaDecision.REJECT_NON_SIGNIFICANT,
                "reason": f"Alpha is not statistically significant (p={reg_result.alpha_pvalue:.4f} > {self.config.p_value_threshold})",
                "suggested_action": "Avoid / Watchlist",
                "alpha": reg_result.alpha,
                "ir": reg_result.ir,
                "pvalue": reg_result.alpha_pvalue
            }

        if reg_result.ir < self.config.ir_reject_threshold:
            return {
                "passed": False,
                "decision": AlphaDecision.REJECT_LOW_IR,
                "reason": f"Information Ratio too low (IR={reg_result.ir:.4f} < {self.config.ir_reject_threshold})",
                "suggested_action": "Reject - Alpha not economically meaningful",
                "alpha": reg_result.alpha,
                "ir": reg_result.ir,
                "pvalue": reg_result.alpha_pvalue
            }

        if gmm_result and not gmm_result.get("is_persistent", True):
            return {
                "passed": False,
                "decision": AlphaDecision.REJECT_NON_PERSISTENT,
                "reason": "Alpha lacks persistence across sub-periods in GMM check",
                "suggested_action": "Watchlist / Downgrade conviction",
                "alpha": reg_result.alpha,
                "ir": reg_result.ir,
                "pvalue": reg_result.alpha_pvalue
            }

        if reg_result.ir >= self.config.ir_attractive_threshold:
            decision = AlphaDecision.HIGH_CONVICTION
            action = "High-conviction opportunity"
        elif reg_result.ir >= self.config.ir_interesting_threshold:
            decision = AlphaDecision.ATTRACTIVE
            action = "Attractive but requires monitoring"
        else:
            decision = AlphaDecision.INTERESTING_SMALL
            action = "Interesting, small position only"

        return {
            "passed": True,
            "decision": decision,
            "reason": f"Alpha statistically (p={reg_result.alpha_pvalue:.4f}) and economically (IR={reg_result.ir:.4f}) verified",
            "suggested_action": action,
            "alpha": reg_result.alpha,
            "ir": reg_result.ir,
            "pvalue": reg_result.alpha_pvalue
        }
