"""
A/B Testing framework with two-proportion z-test and confidence interval
estimation at configurable significance levels (default 95 % CI).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import CONFIDENCE_LEVEL, MIN_SAMPLE_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Container for the outcome of a single A/B test."""

    experiment_id: str
    n_control: int
    n_treatment: int
    conversion_rate_control: float
    conversion_rate_treatment: float
    absolute_lift: float
    relative_lift: float
    z_statistic: float
    p_value: float
    confidence_level: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    power: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "experiment_id": self.experiment_id,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "conversion_rate_control": round(self.conversion_rate_control, 6),
            "conversion_rate_treatment": round(self.conversion_rate_treatment, 6),
            "absolute_lift": round(self.absolute_lift, 6),
            "relative_lift_pct": round(self.relative_lift * 100, 2),
            "z_statistic": round(self.z_statistic, 4),
            "p_value": round(self.p_value, 6),
            "confidence_level_pct": round(self.confidence_level * 100, 1),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "is_significant": self.is_significant,
            "observed_power": round(self.power, 4),
        }


class ABTestingFramework:
    """
    Controlled A/B experiment analysis using a two-tailed two-proportion z-test.

    Supports:
    - Pre-experiment sample size calculation.
    - Conversion rate comparison with CI estimation.
    - Power analysis for the observed effect size.
    - Segment-level subgroup analysis.
    """

    def __init__(self, confidence_level: float = CONFIDENCE_LEVEL) -> None:
        self.confidence_level = confidence_level
        self._alpha = 1.0 - confidence_level
        self._z_critical = float(stats.norm.ppf(1.0 - self._alpha / 2))

    # ------------------------------------------------------------------
    # sample size planning
    # ------------------------------------------------------------------

    def required_sample_size(
        self,
        baseline_rate: float,
        min_detectable_effect: float,
        power: float = 0.80,
    ) -> int:
        """
        Compute the required per-arm sample size for a two-proportion z-test.

        Parameters
        ----------
        baseline_rate : float
            Expected conversion rate in the control group.
        min_detectable_effect : float
            Absolute difference we want to detect.
        power : float
            Desired statistical power (1 − β).

        Returns
        -------
        int : minimum observations per arm.
        """
        p1 = baseline_rate
        p2 = baseline_rate + min_detectable_effect
        p_avg = (p1 + p2) / 2

        z_alpha = self._z_critical
        z_beta = float(stats.norm.ppf(power))

        n = (
            (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
             + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
            / (p2 - p1) ** 2
        )
        n_ceil = int(np.ceil(n))
        logger.info(
            "Required sample size per arm: %d (baseline=%.3f, MDE=%.3f, power=%.2f)",
            n_ceil, baseline_rate, min_detectable_effect, power,
        )
        return n_ceil

    # ------------------------------------------------------------------
    # core test
    # ------------------------------------------------------------------

    def run_test(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment",
        outcome_col: str = "converted",
        experiment_id: str = "exp_001",
    ) -> ABTestResult:
        """
        Run the two-tailed proportion z-test on experiment data.

        Parameters
        ----------
        df : pd.DataFrame
            Experiment log with at least ``treatment_col`` and ``outcome_col``.
        treatment_col : str
            Column indicating treatment (1) vs control (0).
        outcome_col : str
            Binary conversion column.
        experiment_id : str
            Identifier for logging and reporting.

        Returns
        -------
        ABTestResult
        """
        ctrl = df[df[treatment_col] == 0][outcome_col]
        treat = df[df[treatment_col] == 1][outcome_col]

        n_ctrl = len(ctrl)
        n_treat = len(treat)

        if min(n_ctrl, n_treat) < MIN_SAMPLE_SIZE:
            logger.warning(
                "Sample size below minimum (%d). Results may be unreliable.",
                MIN_SAMPLE_SIZE,
            )

        cr_ctrl = float(ctrl.mean())
        cr_treat = float(treat.mean())
        abs_lift = cr_treat - cr_ctrl
        rel_lift = abs_lift / cr_ctrl if cr_ctrl > 0 else 0.0

        # pooled standard error (null hypothesis: equal rates)
        p_pool = (ctrl.sum() + treat.sum()) / (n_ctrl + n_treat)
        se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n_ctrl + 1.0 / n_treat))
        z_stat = abs_lift / se if se > 0 else 0.0
        p_val = float(2 * stats.norm.sf(abs(z_stat)))

        # confidence interval on the difference (unpooled SE)
        se_diff = np.sqrt(
            cr_ctrl * (1 - cr_ctrl) / n_ctrl
            + cr_treat * (1 - cr_treat) / n_treat
        )
        margin = self._z_critical * se_diff
        ci_lower = abs_lift - margin
        ci_upper = abs_lift + margin

        is_sig = p_val < self._alpha
        power = self._compute_power(cr_ctrl, cr_treat, n_ctrl, n_treat)

        result = ABTestResult(
            experiment_id=experiment_id,
            n_control=n_ctrl,
            n_treatment=n_treat,
            conversion_rate_control=cr_ctrl,
            conversion_rate_treatment=cr_treat,
            absolute_lift=abs_lift,
            relative_lift=rel_lift,
            z_statistic=z_stat,
            p_value=p_val,
            confidence_level=self.confidence_level,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_significant=is_sig,
            power=power,
        )

        logger.info(
            "A/B Test [%s] — CR ctrl=%.4f, treat=%.4f | lift=%.4f (%.1f%%) "
            "| z=%.3f, p=%.4f | significant=%s",
            experiment_id,
            cr_ctrl, cr_treat,
            abs_lift, rel_lift * 100,
            z_stat, p_val,
            is_sig,
        )
        return result

    # ------------------------------------------------------------------
    # subgroup / segment analysis
    # ------------------------------------------------------------------

    def segment_analysis(
        self,
        df: pd.DataFrame,
        segment_col: str,
        treatment_col: str = "treatment",
        outcome_col: str = "converted",
        experiment_id: str = "exp_001",
    ) -> pd.DataFrame:
        """
        Run per-segment A/B tests and return a summary DataFrame.

        Parameters
        ----------
        segment_col : str
            Column to segment on (e.g. "segment_encoded", "region_encoded").

        Returns
        -------
        pd.DataFrame with one row per segment, columns from ABTestResult.to_dict().
        """
        records = []
        for seg_val in sorted(df[segment_col].unique()):
            seg_df = df[df[segment_col] == seg_val].copy()
            if seg_df[treatment_col].nunique() < 2:
                continue
            seg_id = f"{experiment_id}_seg{seg_val}"
            try:
                result = self.run_test(seg_df, treatment_col, outcome_col, seg_id)
                row = result.to_dict()
                row["segment"] = seg_val
                records.append(row)
            except ZeroDivisionError:
                logger.warning("Skipping segment %s — division by zero.", seg_val)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------

    def _compute_power(
        self,
        p1: float,
        p2: float,
        n1: int,
        n2: int,
    ) -> float:
        """Estimate observed statistical power for the measured effect."""
        if p1 == p2 or n1 == 0 or n2 == 0:
            return 0.0
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        ncp = abs(p2 - p1) / se
        power = float(stats.norm.sf(self._z_critical - ncp)
                      + stats.norm.cdf(-self._z_critical - ncp))
        return min(power, 1.0)
