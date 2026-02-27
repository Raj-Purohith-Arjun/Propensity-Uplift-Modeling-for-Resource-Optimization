"""
Uplift evaluation metrics: Qini curve, AUUC, incremental lift,
and cumulative gain curve for prioritised resource allocation.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validate_inputs(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> None:
    if len(y) != len(treatment) or len(y) != len(uplift_scores):
        raise ValueError("y, treatment, and uplift_scores must have the same length.")
    if set(np.unique(treatment)) - {0, 1}:
        raise ValueError("treatment must be binary (0/1).")


# ---------------------------------------------------------------------------
# Qini curve
# ---------------------------------------------------------------------------

def qini_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve.

    Ranks observations by descending uplift score and tracks the incremental
    number of conversions attributable to treatment at each targeting depth.

    Parameters
    ----------
    y : binary outcome array.
    treatment : binary treatment indicator.
    uplift_scores : predicted CATE / uplift scores.
    normalize : bool
        If True, x-axis is expressed as a fraction of the population.

    Returns
    -------
    (x, qini_values) both as np.ndarray of length n_samples + 1.
    """
    _validate_inputs(y, treatment, uplift_scores)

    n = len(y)
    order = np.argsort(-uplift_scores)
    y_sorted = y[order]
    t_sorted = treatment[order]

    n_treat_cumsum = np.cumsum(t_sorted)
    n_ctrl_cumsum = np.cumsum(1 - t_sorted)

    with np.errstate(divide="ignore", invalid="ignore"):
        conversions_treat = np.cumsum(y_sorted * t_sorted)
        conversions_ctrl = np.cumsum(y_sorted * (1 - t_sorted))

        n_treat_total = t_sorted.sum()
        n_ctrl_total = (1 - t_sorted).sum()

        # incremental conversions normalised by control size
        qini = np.where(
            n_ctrl_cumsum > 0,
            conversions_treat
            - conversions_ctrl * np.where(n_ctrl_cumsum > 0, n_treat_cumsum / n_ctrl_cumsum, 0),
            0.0,
        )

    qini = np.concatenate([[0.0], qini])
    x = np.arange(n + 1)

    if normalize:
        x = x / n

    return x, qini


def area_under_uplift_curve(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """
    Compute the Area Under the Uplift Curve (AUUC).

    The AUUC is the integral of the Qini curve and measures the overall
    quality of the uplift ranking.
    """
    x, qini = qini_curve(y, treatment, uplift_scores, normalize=True)
    auuc = float(np.trapezoid(qini, x))
    logger.info("AUUC: %.6f", auuc)
    return auuc


# ---------------------------------------------------------------------------
# Incremental lift
# ---------------------------------------------------------------------------

def incremental_lift(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    top_k_fraction: float = 0.30,
) -> Dict[str, float]:
    """
    Measure incremental conversion lift when targeting the top-k% by uplift score.

    Parameters
    ----------
    top_k_fraction : float
        Fraction of population to target (e.g. 0.30 for top 30 %).

    Returns
    -------
    dict with keys:
        incremental_conversions, baseline_conversions, lift_absolute,
        lift_relative_pct, low_value_effort_reduction_pct
    """
    _validate_inputs(y, treatment, uplift_scores)
    n = len(y)
    k = max(1, int(n * top_k_fraction))

    order = np.argsort(-uplift_scores)
    y_sorted = y[order]
    t_sorted = treatment[order]

    top_k_treat = t_sorted[:k]
    top_k_y = y_sorted[:k]

    n_treat_top = top_k_treat.sum()
    n_ctrl_top = k - n_treat_top

    conv_treat_top = (top_k_y * top_k_treat).sum()
    conv_ctrl_top = (top_k_y * (1 - top_k_treat)).sum()

    # overall control conversion rate
    cr_ctrl_global = y[treatment == 0].mean()
    baseline_conversions = cr_ctrl_global * n_treat_top if n_treat_top > 0 else 0.0

    cr_treat_top = conv_treat_top / n_treat_top if n_treat_top > 0 else 0.0
    incremental = conv_treat_top - baseline_conversions

    # low-value effort reduction: fraction of bottom (1 − k) that is NOT targeted
    not_targeted = n - k
    low_value_reduction = not_targeted / n

    result = {
        "top_k_fraction": top_k_fraction,
        "n_targeted": k,
        "cr_treatment_top_k": float(cr_treat_top),
        "cr_control_global": float(cr_ctrl_global),
        "incremental_conversions": float(incremental),
        "baseline_conversions": float(baseline_conversions),
        "lift_absolute": float(cr_treat_top - cr_ctrl_global),
        "lift_relative_pct": float(
            (cr_treat_top - cr_ctrl_global) / cr_ctrl_global * 100
            if cr_ctrl_global > 0 else 0.0
        ),
        "low_value_effort_reduction_pct": float(low_value_reduction * 100),
    }
    logger.info(
        "Incremental lift (top %.0f%%) — absolute: %.4f, relative: %.1f%%, "
        "low-value effort reduction: %.1f%%",
        top_k_fraction * 100,
        result["lift_absolute"],
        result["lift_relative_pct"],
        result["low_value_effort_reduction_pct"],
    )
    return result


# ---------------------------------------------------------------------------
# Cumulative gain curve (for propensity model validation)
# ---------------------------------------------------------------------------

def cumulative_gain_curve(
    y: np.ndarray,
    scores: np.ndarray,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the cumulative gain (response) curve for a propensity model.

    Parameters
    ----------
    y : binary outcome.
    scores : predicted probabilities (higher = more likely to convert).
    normalize : express x as fraction of population.

    Returns
    -------
    (x, cumulative_gain)
    """
    if len(y) != len(scores):
        raise ValueError("y and scores must have the same length.")

    order = np.argsort(-scores)
    y_sorted = y[order]

    cumulative_conversions = np.cumsum(y_sorted)
    total_conversions = y.sum()

    gain = np.concatenate([[0.0], cumulative_conversions / total_conversions])
    x = np.arange(len(y) + 1)

    if normalize:
        x = x / len(y)

    return x, gain


def compute_all_metrics(
    y: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    propensity_scores: np.ndarray,
    top_k_fraction: float = 0.30,
) -> Dict[str, float]:
    """
    Convenience wrapper: compute AUUC, incremental lift, and Qini.

    Returns
    -------
    Flat dict of all evaluation metrics.
    """
    auuc = area_under_uplift_curve(y, treatment, uplift_scores)
    lift_metrics = incremental_lift(y, treatment, uplift_scores, top_k_fraction)

    _, qini_vals = qini_curve(y, treatment, uplift_scores, normalize=True)
    qini_max = float(qini_vals.max())

    from sklearn.metrics import roc_auc_score
    auc_roc = float(roc_auc_score(y, propensity_scores))

    return {
        "auuc": auuc,
        "qini_max": qini_max,
        "propensity_roc_auc": auc_roc,
        **lift_metrics,
    }
