"""
General utility helpers: logging configuration, reproducibility seeding,
and experiment-level summary formatting.
"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """
    Configure root logger with a consistent format.

    Parameters
    ----------
    level : int
        Python logging level (default INFO).
    log_file : str | None
        Optional file path to write logs. Logs are always echoed to stdout.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def seed_everything(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_results(results: Dict[str, Any], path: str) -> None:
    """Persist a flat metric dictionary as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results["_saved_at"] = datetime.utcnow().isoformat()
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logging.getLogger(__name__).info("Results saved to %s", path)


def format_experiment_summary(
    ab_result_dict: Dict[str, Any],
    uplift_metrics: Dict[str, float],
    allocation_summary: Dict[str, float],
) -> str:
    """
    Format a human-readable experiment report.

    Parameters
    ----------
    ab_result_dict : dict from ABTestResult.to_dict().
    uplift_metrics : dict from compute_all_metrics().
    allocation_summary : dict from ResourceAllocator.allocation_summary().

    Returns
    -------
    str : multi-line formatted report.
    """
    lines = [
        "=" * 70,
        "  EXPERIMENT SUMMARY",
        "=" * 70,
        "",
        "[A/B Test Results]",
        f"  Experiment ID       : {ab_result_dict.get('experiment_id', 'N/A')}",
        f"  Control  n          : {ab_result_dict.get('n_control', 0):,}",
        f"  Treatment n         : {ab_result_dict.get('n_treatment', 0):,}",
        f"  Control conv. rate  : {ab_result_dict.get('conversion_rate_control', 0):.4f}",
        f"  Treatment conv. rate: {ab_result_dict.get('conversion_rate_treatment', 0):.4f}",
        f"  Absolute lift       : {ab_result_dict.get('absolute_lift', 0):.4f}",
        f"  Relative lift       : {ab_result_dict.get('relative_lift_pct', 0):.1f}%",
        f"  95% CI              : [{ab_result_dict.get('ci_lower', 0):.4f}, "
        f"{ab_result_dict.get('ci_upper', 0):.4f}]",
        f"  p-value             : {ab_result_dict.get('p_value', 1):.6f}",
        f"  Statistically sig.  : {ab_result_dict.get('is_significant', False)}",
        f"  Observed power      : {ab_result_dict.get('observed_power', 0):.4f}",
        "",
        "[Uplift Model Metrics]",
        f"  AUUC                : {uplift_metrics.get('auuc', 0):.6f}",
        f"  Qini max            : {uplift_metrics.get('qini_max', 0):.4f}",
        f"  Propensity ROC-AUC  : {uplift_metrics.get('propensity_roc_auc', 0):.4f}",
        f"  Incremental lift    : {uplift_metrics.get('lift_relative_pct', 0):.1f}%",
        f"  Low-val effort red. : {uplift_metrics.get('low_value_effort_reduction_pct', 0):.1f}%",
        "",
        "[Resource Allocation Summary]",
        f"  Total customers     : {allocation_summary.get('n_total', 0):,}",
        f"  Targeted            : {allocation_summary.get('n_targeted', 0):,} "
        f"({allocation_summary.get('targeting_rate_pct', 0):.1f}%)",
        f"  High-value tier     : {allocation_summary.get('n_high_value', 0):,}",
        f"  Medium-value tier   : {allocation_summary.get('n_medium_value', 0):,}",
        f"  Low-value tier      : {allocation_summary.get('n_low_value', 0):,}",
        f"  Avg propensity (targeted): {allocation_summary.get('avg_propensity_targeted', 0):.4f}",
        f"  Avg uplift (targeted)    : {allocation_summary.get('avg_uplift_targeted', 0):.4f}",
        f"  Low-value effort reduction: "
        f"{allocation_summary.get('low_value_effort_reduction_pct', 0):.1f}%",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)
