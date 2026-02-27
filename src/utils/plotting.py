"""
Visualisation utilities: Qini curve, cumulative gain, feature importance,
propensity distribution, and allocation tier plots.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PALETTE = sns.color_palette("deep")
_FIGSIZE_WIDE = (12, 5)
_FIGSIZE_SQUARE = (8, 6)


def plot_qini_curve(
    x: np.ndarray,
    qini: np.ndarray,
    title: str = "Qini Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the Qini uplift curve with a random-targeting baseline."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    ax.plot(x, qini, color=_PALETTE[0], linewidth=2, label="Uplift Model")
    ax.axhline(0, color="grey", linestyle="--", linewidth=1, label="Random Baseline")

    ax.fill_between(x, 0, qini, alpha=0.15, color=_PALETTE[0])

    ax.set_xlabel("Fraction of Population Targeted", fontsize=12)
    ax.set_ylabel("Incremental Conversions (Qini)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Qini curve saved to %s", save_path)
    return fig


def plot_cumulative_gain(
    x: np.ndarray,
    gain: np.ndarray,
    title: str = "Cumulative Gain Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cumulative gain with a random baseline diagonal."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    ax.plot(x, gain, color=_PALETTE[1], linewidth=2, label="Propensity Model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random")

    ax.fill_between(x, x, gain, alpha=0.12, color=_PALETTE[1])

    ax.set_xlabel("Fraction of Population Contacted", fontsize=12)
    ax.set_ylabel("Fraction of Conversions Captured", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Cumulative gain curve saved to %s", save_path)
    return fig


def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))

    bars = ax.barh(top.index[::-1], top.values[::-1], color=_PALETTE[2], edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Feature importance plot saved to %s", save_path)
    return fig


def plot_propensity_distribution(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    title: str = "Propensity Score Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlapping KDE of propensity scores by treatment arm."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    treat_scores = propensity_scores[treatment == 1]
    ctrl_scores = propensity_scores[treatment == 0]

    sns.kdeplot(treat_scores, ax=ax, label="Treatment", color=_PALETTE[0], fill=True, alpha=0.4)
    sns.kdeplot(ctrl_scores, ax=ax, label="Control", color=_PALETTE[1], fill=True, alpha=0.4)

    ax.set_xlabel("Propensity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Propensity distribution saved to %s", save_path)
    return fig


def plot_uplift_distribution(
    uplift_scores: np.ndarray,
    title: str = "Uplift Score Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram and KDE of estimated CATE/uplift scores."""
    fig, ax = plt.subplots(figsize=_FIGSIZE_SQUARE)

    sns.histplot(
        uplift_scores, bins=50, kde=True, ax=ax,
        color=_PALETTE[3], edgecolor="white", alpha=0.7,
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1.5, label="No Effect")
    ax.axvline(
        float(np.mean(uplift_scores)), color=_PALETTE[0],
        linestyle="-.", linewidth=1.5,
        label=f"Mean Uplift = {float(np.mean(uplift_scores)):.4f}",
    )

    ax.set_xlabel("Estimated Uplift (CATE)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Uplift distribution saved to %s", save_path)
    return fig


def plot_allocation_tiers(
    allocated_df: pd.DataFrame,
    title: str = "Resource Allocation by Value Tier",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked bar showing targeting decisions vs value tier."""
    tier_order = ["high", "medium", "low"]
    tier_colors = {
        "high": _PALETTE[2],
        "medium": _PALETTE[0],
        "low": _PALETTE[4],
    }

    fig, axes = plt.subplots(1, 2, figsize=_FIGSIZE_WIDE)

    # Left: tier distribution
    tier_counts = allocated_df["value_tier"].value_counts()
    tier_counts = tier_counts.reindex(tier_order, fill_value=0)
    axes[0].bar(
        tier_counts.index,
        tier_counts.values,
        color=[tier_colors[t] for t in tier_counts.index],
        edgecolor="white",
    )
    axes[0].set_title("Customers by Value Tier", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: targeted vs not by tier
    cross = pd.crosstab(allocated_df["value_tier"], allocated_df["is_targeted"])
    cross = cross.reindex(tier_order, fill_value=0)
    cross.columns = ["Not Targeted", "Targeted"]
    cross.plot(
        kind="bar", ax=axes[1], stacked=True,
        color=[_PALETTE[4], _PALETTE[2]], edgecolor="white",
    )
    axes[1].set_title("Targeting Decision by Value Tier", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Value Tier")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Allocation tier plot saved to %s", save_path)
    return fig
