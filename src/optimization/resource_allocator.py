"""
Data-driven resource allocation optimizer.

Combines propensity scores and uplift estimates to rank customers
and prioritise operational resources, maximising incremental conversions
while reducing low-value effort within a given budget constraint.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    HIGH_VALUE_PROPENSITY_THRESHOLD,
    LOW_VALUE_EFFORT_THRESHOLD,
    RESOURCE_BUDGET_FRACTION,
)

logger = logging.getLogger(__name__)


class ResourceAllocator:
    """
    Prioritise and allocate limited operational resources to customers most
    likely to generate incremental value.

    Scoring strategy
    ----------------
    A composite score blends propensity (likelihood to convert) and uplift
    (incremental effect of treatment). High-propensity customers who show
    strong uplift are prioritised. Customers who would convert anyway (high
    propensity, low uplift) and customers with no realistic uplift are
    deprioritised to reduce wasteful effort.

        composite = α × propensity_score + (1 − α) × uplift_score_normalised

    Parameters
    ----------
    budget_fraction : float
        Fraction of the total customer population to contact/intervene.
    propensity_weight : float (α)
        Weight given to propensity vs. uplift in the composite score.
    low_value_threshold : float
        Customers whose normalised composite score is below this percentile
        are classified as low-value and suppressed.
    """

    def __init__(
        self,
        budget_fraction: float = RESOURCE_BUDGET_FRACTION,
        propensity_weight: float = 0.40,
        low_value_threshold: float = LOW_VALUE_EFFORT_THRESHOLD,
    ) -> None:
        if not 0 < budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1].")
        if not 0 <= propensity_weight <= 1:
            raise ValueError("propensity_weight must be in [0, 1].")

        self.budget_fraction = budget_fraction
        self.propensity_weight = propensity_weight
        self.uplift_weight = 1.0 - propensity_weight
        self.low_value_threshold = low_value_threshold

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def compute_composite_score(
        self,
        propensity_scores: np.ndarray,
        uplift_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Blend propensity and (min-max normalised) uplift into a single score.
        """
        uplift_min = uplift_scores.min()
        uplift_max = uplift_scores.max()
        denom = uplift_max - uplift_min
        if denom == 0:
            uplift_norm = np.zeros_like(uplift_scores)
        else:
            uplift_norm = (uplift_scores - uplift_min) / denom

        composite = (
            self.propensity_weight * propensity_scores
            + self.uplift_weight * uplift_norm
        )
        return composite

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        df: pd.DataFrame,
        propensity_col: str = "propensity_score",
        uplift_col: str = "uplift_score",
        customer_id_col: str = "customer_id",
    ) -> pd.DataFrame:
        """
        Rank customers and flag those selected for resource allocation.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``propensity_col``, ``uplift_col``, and ``customer_id_col``.

        Returns
        -------
        pd.DataFrame with additional columns:
            composite_score, rank, is_targeted, value_tier
        """
        out = df.copy()

        composite = self.compute_composite_score(
            out[propensity_col].values,
            out[uplift_col].values,
        )
        out["composite_score"] = composite
        out["rank"] = pd.Series(composite).rank(ascending=False, method="first").astype(int)

        n_total = len(out)
        budget_n = max(1, int(np.ceil(n_total * self.budget_fraction)))
        out["is_targeted"] = out["rank"] <= budget_n

        # low-value: bottom fraction by composite score
        low_cutoff = np.percentile(composite, self.low_value_threshold * 100)
        out["value_tier"] = "medium"
        out.loc[composite >= np.percentile(composite, 70), "value_tier"] = "high"
        out.loc[composite < low_cutoff, "value_tier"] = "low"

        targeted = out["is_targeted"].sum()
        low_value_suppressed = (~out["is_targeted"] & (out["value_tier"] == "low")).sum()
        reduction_pct = low_value_suppressed / n_total * 100

        logger.info(
            "Resource allocation — budget: %.0f%% (%d customers), "
            "low-value suppressed: %d (%.1f%% effort reduction)",
            self.budget_fraction * 100,
            targeted,
            low_value_suppressed,
            reduction_pct,
        )
        return out

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def allocation_summary(self, allocated_df: pd.DataFrame) -> Dict[str, float]:
        """
        Return a summary dictionary of key allocation metrics.
        """
        n = len(allocated_df)
        n_targeted = allocated_df["is_targeted"].sum()
        n_low = (allocated_df["value_tier"] == "low").sum()
        n_high = (allocated_df["value_tier"] == "high").sum()
        n_medium = (allocated_df["value_tier"] == "medium").sum()

        avg_propensity_targeted = (
            allocated_df.loc[allocated_df["is_targeted"], "propensity_score"].mean()
        )
        avg_uplift_targeted = (
            allocated_df.loc[allocated_df["is_targeted"], "uplift_score"].mean()
        )

        not_targeted_low = (
            ~allocated_df["is_targeted"] & (allocated_df["value_tier"] == "low")
        ).sum()
        effort_reduction_pct = not_targeted_low / n * 100

        return {
            "n_total": int(n),
            "n_targeted": int(n_targeted),
            "targeting_rate_pct": round(n_targeted / n * 100, 2),
            "n_high_value": int(n_high),
            "n_medium_value": int(n_medium),
            "n_low_value": int(n_low),
            "avg_propensity_targeted": round(float(avg_propensity_targeted), 4),
            "avg_uplift_targeted": round(float(avg_uplift_targeted), 4),
            "low_value_effort_reduction_pct": round(effort_reduction_pct, 2),
        }
