import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class CrossValidationSignal:
    node_name: str
    target_ticker: str
    peer_ticker: str
    relation: str  # 'upstream_supplier' or 'downstream_customer'
    signal_type: str  # 'RESONANCE' or 'DIVERGENCE'
    confidence: float
    description: str

class SupplyChainCrossValidator:
    """
    Layer 1 Supply Chain Cross-Validator.
    Performs financial cross-validation between target company and upstream/downstream nodes
    (e.g., inventory build-up vs. capex growth vs. prepayments).
    """
    def __init__(self, divergence_threshold: float = 0.25):
        self.divergence_threshold = divergence_threshold

    def validate_node_financials(
        self,
        target_metrics: Dict[str, float],
        downstream_metrics: Dict[str, float],
        upstream_metrics: Optional[Dict[str, float]] = None
    ) -> List[CrossValidationSignal]:
        signals = []

        # 1. Downstream Capex vs. Target Revenue/Order Growth
        target_rev_growth = target_metrics.get("revenue_growth_yoy", 0.0)
        downstream_capex_growth = downstream_metrics.get("capex_growth_yoy", 0.0)

        diff = abs(target_rev_growth - downstream_capex_growth)
        if diff <= self.divergence_threshold:
            signals.append(CrossValidationSignal(
                node_name="Downstream Capex Resonance",
                target_ticker=target_metrics.get("ticker", "TARGET"),
                peer_ticker=downstream_metrics.get("ticker", "PEER_DOWNSTREAM"),
                relation="downstream_customer",
                signal_type="RESONANCE",
                confidence=max(0.0, 1.0 - diff),
                description=f"Target revenue growth ({target_rev_growth:.1%}) resonates with downstream Capex growth ({downstream_capex_growth:.1%})"
            ))
        else:
            signals.append(CrossValidationSignal(
                node_name="Downstream Capex Divergence",
                target_ticker=target_metrics.get("ticker", "TARGET"),
                peer_ticker=downstream_metrics.get("ticker", "PEER_DOWNSTREAM"),
                relation="downstream_customer",
                signal_type="DIVERGENCE",
                confidence=min(1.0, diff),
                description=f"Divergence detected: Target rev growth ({target_rev_growth:.1%}) vs Downstream capex ({downstream_capex_growth:.1%})"
            ))

        # 2. Upstream Inventory / Prepayments vs. Target COGS / Raw Material Procurement
        if upstream_metrics:
            target_prepayment = target_metrics.get("prepayment_growth_yoy", 0.0)
            upstream_contract_liab = upstream_metrics.get("contract_liabilities_growth_yoy", 0.0)
            diff_up = abs(target_prepayment - upstream_contract_liab)
            if diff_up <= self.divergence_threshold:
                signals.append(CrossValidationSignal(
                    node_name="Upstream Prepayment Resonance",
                    target_ticker=target_metrics.get("ticker", "TARGET"),
                    peer_ticker=upstream_metrics.get("ticker", "PEER_UPSTREAM"),
                    relation="upstream_supplier",
                    signal_type="RESONANCE",
                    confidence=max(0.0, 1.0 - diff_up),
                    description="Target prepayments align with upstream contract liabilities."
                ))
            else:
                signals.append(CrossValidationSignal(
                    node_name="Upstream Prepayment Divergence",
                    target_ticker=target_metrics.get("ticker", "TARGET"),
                    peer_ticker=upstream_metrics.get("ticker", "PEER_UPSTREAM"),
                    relation="upstream_supplier",
                    signal_type="DIVERGENCE",
                    confidence=min(1.0, diff_up),
                    description="Divergence between target prepayments and upstream contract liabilities."
                ))

        return signals
