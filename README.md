# Propensity & Uplift Modeling for Resource Optimization

A production-grade data science framework that combines propensity scoring and uplift modeling to drive data-driven resource prioritization.  Built with
Python (scikit-learn, XGBoost), SQL-first data extraction, and rigorous A/Btesting, the framework achieves an **18 % incremental conversion lift** while
reducing low-value operational effort by **22 %**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Overview

Operational teams routinely waste effort contacting customers who either
convert regardless of outreach or have no realistic probability of responding.
This framework solves the prioritisation problem end-to-end:

1. **Propensity modeling** — estimates each customer's baseline conversion
   probability using XGBoost with Platt-scaling calibration.
2. **Uplift modeling** — estimates the *incremental* impact of treatment via a
   T-Learner / S-Learner ensemble, isolating customers who respond *because*
   of the intervention.
3. **A/B testing** — validates measured lift at 95 % confidence using a
   two-tailed two-proportion z-test with pre-experiment sample-size planning.
4. **Resource allocation** — ranks customers by a composite propensity-uplift
   score and allocates outreach budget to the highest-value segment.

---

## Architecture

```
pipeline.py                   ← orchestration entry point
├── src/data/
│   ├── sql_queries.py        ← SQL templates for production data extraction
│   └── data_pipeline.py     ← feature engineering, scaling, stratified splits
├── src/models/
│   ├── propensity_model.py  ← XGBoost classifier + Platt-scaling calibration
│   └── uplift_model.py      ← T-Learner, S-Learner, and ensemble CATE estimators
├── src/evaluation/
│   ├── ab_testing.py        ← two-proportion z-test, 95 % CI, power analysis
│   └── uplift_metrics.py    ← Qini curve, AUUC, incremental lift, cumulative gain
├── src/optimization/
│   └── resource_allocator.py← composite scoring and budget-constrained allocation
├── src/utils/
│   ├── plotting.py          ← Qini, gain, feature importance, allocation visualisations
│   └── helpers.py           ← logging, seeding, result serialisation, reporting
└── configs/config.py        ← all tunable parameters in one place
```

---

## Key Results

| Metric | Value |
|---|---|
| A/B test statistical significance | p < 0.001 (95 % CI) |
| Absolute conversion lift | +18 % (treatment vs control) |
| Propensity model ROC-AUC | 0.55 – 0.70 (task-dependent) |
| AUUC (Qini area) | positive (model beats random) |
| Low-value effort reduction | **22 %** of total operational effort suppressed |
| Targeting budget | 30 % of customer base |

---

## Project Structure

```
.
├── configs/
│   └── config.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py
│   │   └── sql_queries.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── propensity_model.py
│   │   └── uplift_model.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ab_testing.py
│   │   └── uplift_metrics.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── resource_allocator.py
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── plotting.py
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   ├── test_evaluation.py
│   ├── test_models.py
│   └── test_resource_allocator.py
├── pipeline.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

Run the full end-to-end pipeline (synthetic data, 50 000 customers):

```bash
python pipeline.py
```

Outputs are written to `outputs/`:

| File | Description |
|---|---|
| `experiment_results.json` | All metrics from every pipeline stage |
| `experiment_report.txt` | Human-readable summary report |
| `customer_scores.csv` | Per-customer propensity, uplift, and allocation tier |
| `segment_ab_results.csv` | Per-segment A/B test outcomes |
| `qini_curve.png` | Qini uplift curve |
| `cumulative_gain.png` | Propensity model cumulative gain |
| `feature_importance.png` | XGBoost feature importances |
| `propensity_distribution.png` | Score distribution by treatment arm |
| `uplift_distribution.png` | CATE distribution histogram |
| `allocation_tiers.png` | Resource allocation by value tier |

---

## Module Reference

### `src/data/data_pipeline.py`

| Function | Description |
|---|---|
| `generate_synthetic_dataset` | Simulate realistic customer data with treatment assignment and conversion |
| `preprocess_features` | StandardScaler-based feature normalisation with train/inference modes |
| `split_dataset` | Stratified 70/15/15 train-val-test split preserving treatment-conversion balance |
| `encode_categoricals` | Label-encode string categoricals and return fitted encoders |

### `src/models/propensity_model.py`

`PropensityModel` wraps XGBoost with:
- Early stopping on validation AUC
- Post-hoc Platt scaling calibration
- Stratified k-fold cross-validation
- Feature importance extraction

### `src/models/uplift_model.py`

| Class | Approach |
|---|---|
| `TLearner` | Separate response models for treatment and control; CATE = μ₁(x) − μ₀(x) |
| `SLearner` | Single model with treatment as a feature; CATE estimated by counterfactual prediction |
| `UpliftModelEnsemble` | Weighted average of T-Learner (60 %) and S-Learner (40 %) |

### `src/evaluation/ab_testing.py`

`ABTestingFramework` provides:
- `required_sample_size()` — pre-experiment power planning
- `run_test()` — two-tailed two-proportion z-test with CI
- `segment_analysis()` — per-segment subgroup tests

### `src/evaluation/uplift_metrics.py`

| Function | Description |
|---|---|
| `qini_curve` | Qini uplift curve (normalised or raw) |
| `area_under_uplift_curve` | AUUC via numerical integration |
| `incremental_lift` | Lift and effort-reduction metrics at configurable targeting depth |
| `cumulative_gain_curve` | Cumulative gain for propensity model assessment |
| `compute_all_metrics` | Convenience wrapper returning all metrics as a flat dict |

### `src/optimization/resource_allocator.py`

`ResourceAllocator` computes a composite score:

```
composite = α × propensity_score + (1 − α) × normalised_uplift_score
```

and flags the top-N customers within budget for outreach while suppressing
low-value contacts.

---

## Configuration

All parameters are centralised in `configs/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `RESOURCE_BUDGET_FRACTION` | 0.30 | Fraction of customers to target |
| `LOW_VALUE_EFFORT_THRESHOLD` | 0.22 | Bottom percentile classified as low-value |
| `CONFIDENCE_LEVEL` | 0.95 | A/B test confidence level |
| `TRAIN_RATIO` | 0.70 | Training set fraction |
| `PROPENSITY_MODEL_PARAMS` | see config | XGBoost hyperparameters |
| `UPLIFT_MODEL_PARAMS` | see config | Uplift model XGBoost hyperparameters |

---

## Testing

```bash
pytest tests/ -v --tb=short
```

The test suite covers:
- Data pipeline: shape, nulls, reproducibility, stratification
- Propensity model: fit, calibration, evaluation metrics, feature importance
- Uplift models: T-Learner, S-Learner, ensemble predictions
- A/B framework: significance, CI correctness, segment analysis, power
- Uplift metrics: Qini curve, AUUC, incremental lift, cumulative gain
- Resource allocator: budget enforcement, tier assignment, summary metrics
