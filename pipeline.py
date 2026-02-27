"""
End-to-end pipeline runner.

Orchestrates:
  1. Data generation / loading
  2. Feature preprocessing and splitting
  3. Propensity model training and evaluation
  4. Uplift model training (T-Learner + S-Learner ensemble)
  5. A/B testing and hypothesis testing (95% CI)
  6. Resource allocation and optimisation
  7. Metric aggregation and reporting
"""
from __future__ import annotations

import logging
import os
import sys

# ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from configs.config import (
    ALL_FEATURES,
    RANDOM_STATE,
    RESOURCE_BUDGET_FRACTION,
    TARGET_COLUMN,
    TREATMENT_COLUMN,
)
from src.data.data_pipeline import (
    generate_synthetic_dataset,
    preprocess_features,
    split_dataset,
)
from src.evaluation.ab_testing import ABTestingFramework
from src.evaluation.uplift_metrics import compute_all_metrics, qini_curve, cumulative_gain_curve
from src.models.propensity_model import PropensityModel
from src.models.uplift_model import UpliftModelEnsemble
from src.optimization.resource_allocator import ResourceAllocator
from src.utils.helpers import format_experiment_summary, save_results, seed_everything, setup_logging

logger = logging.getLogger(__name__)


def run_pipeline(
    n_samples: int = 50000,
    output_dir: str = "outputs",
    experiment_id: str = "exp_applied_materials_001",
) -> dict:
    """
    Execute the full propensity & uplift modelling pipeline.

    Parameters
    ----------
    n_samples : int
        Number of synthetic customer records to generate.
    output_dir : str
        Directory for saving results and plots.
    experiment_id : str
        Unique identifier for the experiment run.

    Returns
    -------
    dict : consolidated metrics from all pipeline stages.
    """
    setup_logging()
    seed_everything(RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. data
    # ------------------------------------------------------------------
    logger.info("Generating synthetic dataset (%d samples)…", n_samples)
    df = generate_synthetic_dataset(n_samples=n_samples, random_state=RANDOM_STATE)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        t_train, t_val, t_test,
    ) = split_dataset(df)

    X_train_scaled, scaler = preprocess_features(X_train, fit_scaler=True)
    X_val_scaled, _ = preprocess_features(X_val, scaler=scaler, fit_scaler=False)
    X_test_scaled, _ = preprocess_features(X_test, scaler=scaler, fit_scaler=False)

    # ------------------------------------------------------------------
    # 2. propensity model
    # ------------------------------------------------------------------
    logger.info("Training propensity model…")
    prop_model = PropensityModel()
    prop_model.fit(X_train_scaled, y_train, X_val_scaled, y_val, calibrate=True)

    prop_metrics = prop_model.evaluate(X_test_scaled, y_test)
    propensity_scores_test = prop_model.predict_proba(X_test_scaled)

    # ------------------------------------------------------------------
    # 3. uplift model
    # ------------------------------------------------------------------
    logger.info("Training uplift ensemble (T-Learner + S-Learner)…")
    uplift_model = UpliftModelEnsemble()
    uplift_model.fit(X_train_scaled, y_train, t_train)

    uplift_scores_test = uplift_model.predict_uplift(X_test_scaled)

    # ------------------------------------------------------------------
    # 4. uplift evaluation
    # ------------------------------------------------------------------
    uplift_metrics = compute_all_metrics(
        y_test.values,
        t_test.values,
        uplift_scores_test,
        propensity_scores_test,
        top_k_fraction=RESOURCE_BUDGET_FRACTION,
    )

    # ------------------------------------------------------------------
    # 5. A/B testing (95% CI)
    # ------------------------------------------------------------------
    logger.info("Running A/B hypothesis test (95%% CI)…")
    ab_framework = ABTestingFramework(confidence_level=0.95)

    test_df = X_test_scaled.copy()
    test_df[TREATMENT_COLUMN] = t_test.values
    test_df[TARGET_COLUMN] = y_test.values

    ab_result = ab_framework.run_test(
        test_df,
        treatment_col=TREATMENT_COLUMN,
        outcome_col=TARGET_COLUMN,
        experiment_id=experiment_id,
    )
    ab_dict = ab_result.to_dict()

    # segment-level A/B analysis
    if "segment_encoded" in test_df.columns:
        seg_results = ab_framework.segment_analysis(
            test_df,
            segment_col="segment_encoded",
            treatment_col=TREATMENT_COLUMN,
            outcome_col=TARGET_COLUMN,
            experiment_id=experiment_id,
        )
        seg_results.to_csv(
            os.path.join(output_dir, "segment_ab_results.csv"), index=False
        )
        logger.info("Segment A/B results saved.")

    # ------------------------------------------------------------------
    # 6. resource allocation
    # ------------------------------------------------------------------
    logger.info("Running resource allocation optimisation…")
    allocator = ResourceAllocator(budget_fraction=RESOURCE_BUDGET_FRACTION)

    score_df = pd.DataFrame(
        {
            "customer_id": df.iloc[y_test.index]["customer_id"].values,
            "propensity_score": propensity_scores_test,
            "uplift_score": uplift_scores_test,
        }
    )
    allocated_df = allocator.allocate(score_df)
    alloc_summary = allocator.allocation_summary(allocated_df)

    allocated_df.to_csv(
        os.path.join(output_dir, "customer_scores.csv"), index=False
    )

    # ------------------------------------------------------------------
    # 7. plots (non-blocking; saved to disk)
    # ------------------------------------------------------------------
    try:
        from src.utils.plotting import (
            plot_qini_curve,
            plot_cumulative_gain,
            plot_feature_importance,
            plot_propensity_distribution,
            plot_uplift_distribution,
            plot_allocation_tiers,
        )

        x_qini, qini_vals = qini_curve(
            y_test.values, t_test.values, uplift_scores_test
        )
        plot_qini_curve(
            x_qini, qini_vals,
            save_path=os.path.join(output_dir, "qini_curve.png"),
        )

        x_gain, gain_vals = cumulative_gain_curve(y_test.values, propensity_scores_test)
        plot_cumulative_gain(
            x_gain, gain_vals,
            save_path=os.path.join(output_dir, "cumulative_gain.png"),
        )

        plot_feature_importance(
            prop_model.get_feature_importance(),
            save_path=os.path.join(output_dir, "feature_importance.png"),
        )

        plot_propensity_distribution(
            propensity_scores_test, t_test.values,
            save_path=os.path.join(output_dir, "propensity_distribution.png"),
        )

        plot_uplift_distribution(
            uplift_scores_test,
            save_path=os.path.join(output_dir, "uplift_distribution.png"),
        )

        plot_allocation_tiers(
            allocated_df,
            save_path=os.path.join(output_dir, "allocation_tiers.png"),
        )
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    # ------------------------------------------------------------------
    # 8. summary
    # ------------------------------------------------------------------
    all_metrics = {
        "experiment_id": experiment_id,
        "propensity_metrics": prop_metrics,
        "uplift_metrics": uplift_metrics,
        "ab_test": ab_dict,
        "allocation": alloc_summary,
    }

    save_results(all_metrics, os.path.join(output_dir, "experiment_results.json"))

    report = format_experiment_summary(ab_dict, uplift_metrics, alloc_summary)
    print(report)

    results_path = os.path.join(output_dir, "experiment_report.txt")
    with open(results_path, "w") as fh:
        fh.write(report)

    return all_metrics


if __name__ == "__main__":
    run_pipeline()
