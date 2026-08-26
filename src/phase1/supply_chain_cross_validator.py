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

    def validate_financial_statements_df(
        self,
        target_df: pd.DataFrame,
        downstream_df: pd.DataFrame,
        upstream_df: Optional[pd.DataFrame] = None
    ) -> List[CrossValidationSignal]:
        """
        Batch cross-validation using real financial statement DataFrames (A-Share & US filings).
        Mapped standard accounting items:
        - target: 'revenue_growth_yoy' (营业收入同比增长), 'prepayment_growth_yoy' (预付款项同比增长), 'inventory_growth_yoy' (存货同比增长)
        - downstream: 'capex_growth_yoy' (购建固定资产支付的现金同比增长), 'revenue_growth_yoy'
        - upstream: 'contract_liabilities_growth_yoy' (合同负债同比增长), 'inventory_growth_yoy'
        """
        target_metrics = {
            "ticker": str(target_df.get("ticker", "TARGET").iloc[-1]) if "ticker" in target_df else "TARGET",
            "revenue_growth_yoy": float(target_df.get("revenue_growth_yoy", pd.Series([0.0])).iloc[-1]),
            "prepayment_growth_yoy": float(target_df.get("prepayment_growth_yoy", pd.Series([0.0])).iloc[-1]),
            "inventory_growth_yoy": float(target_df.get("inventory_growth_yoy", pd.Series([0.0])).iloc[-1])
        }

        downstream_metrics = {
            "ticker": str(downstream_df.get("ticker", "DOWNSTREAM").iloc[-1]) if "ticker" in downstream_df else "DOWNSTREAM",
            "capex_growth_yoy": float(downstream_df.get("capex_growth_yoy", pd.Series([0.0])).iloc[-1]),
            "revenue_growth_yoy": float(downstream_df.get("revenue_growth_yoy", pd.Series([0.0])).iloc[-1])
        }

        upstream_metrics = None
        if upstream_df is not None:
            upstream_metrics = {
                "ticker": str(upstream_df.get("ticker", "UPSTREAM").iloc[-1]) if "ticker" in upstream_df else "UPSTREAM",
                "contract_liabilities_growth_yoy": float(upstream_df.get("contract_liabilities_growth_yoy", pd.Series([0.0])).iloc[-1]),
                "inventory_growth_yoy": float(upstream_df.get("inventory_growth_yoy", pd.Series([0.0])).iloc[-1])
            }

        return self.validate_node_financials(target_metrics, downstream_metrics, upstream_metrics)

