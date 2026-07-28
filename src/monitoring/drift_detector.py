import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List


class StatisticalDriftDetector:
    """Monitors feature distributions for statistical data drift using Kolmogorov-Smirnov test."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        drift_results = {}
        drift_detected_cols = []

        feature_cols = [c for c in reference_df.columns if c != "target"]

        for col in feature_cols:
            ref_data = reference_df[col].dropna()
            curr_data = current_df[col].dropna()

            ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
            is_drift = bool(p_value < self.alpha)

            if is_drift:
                drift_detected_cols.append(col)

            drift_results[col] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": is_drift
            }

        overall_drift = len(drift_detected_cols) > 0

        return {
            "overall_drift_detected": overall_drift,
            "drifted_features": drift_detected_cols,
            "feature_details": drift_results
        }
