import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CalibrationMetrics:
    brier_score: float
    cross_entropy: float
    accuracy: float
    calibration_error: float

class CalibrationRLSPLoop:
    """
    Layer 4 Brier Score & Cross-Entropy Calibration Loop + KNN Pattern Matching for LLM prompts.
    """
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    def compute_brier_score(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """
        Brier Score = 1/N * sum((p_i - o_i)^2)
        Lower is better.
        """
        return float(np.mean((predicted_probs - actual_outcomes) ** 2))

    def compute_cross_entropy(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray, eps: float = 1e-12) -> float:
        """
        Cross-Entropy Loss = -1/N * sum(y * log(p) + (1-y) * log(1-p))
        """
        p_clipped = np.clip(predicted_probs, eps, 1.0 - eps)
        loss = -np.mean(actual_outcomes * np.log(p_clipped) + (1.0 - actual_outcomes) * np.log(1.0 - p_clipped))
        return float(loss)

    def evaluate_predictions(self, predicted_probs: List[float], actual_outcomes: List[int]) -> CalibrationMetrics:
        p = np.array(predicted_probs, dtype=float)
        y = np.array(actual_outcomes, dtype=int)

        brier = self.compute_brier_score(p, y)
        ce = self.compute_cross_entropy(p, y)
        acc = float(np.mean((p >= 0.5) == (y == 1)))
        cal_err = float(abs(np.mean(p) - np.mean(y)))

        return CalibrationMetrics(
            brier_score=brier,
            cross_entropy=ce,
            accuracy=acc,
            calibration_error=cal_err
        )

    def find_knn_analogues(
        self,
        current_features: np.ndarray,
        historical_database: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Finds k nearest historical analogue market patterns using Euclidean distance.
        """
        if not historical_database:
            return []

        distances = []
        for item in historical_database:
            hist_feat = np.array(item["features"], dtype=float)
            dist = float(np.linalg.norm(current_features - hist_feat))
            distances.append((dist, item))

        distances.sort(key=lambda x: x[0])
        return [item for _, item in distances[:self.n_neighbors]]
