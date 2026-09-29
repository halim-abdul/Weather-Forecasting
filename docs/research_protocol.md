# Research protocol

## Objective

Evaluate machine-learning and deep-learning models for short- to medium-horizon weather forecasting while preventing temporal leakage and quantifying uncertainty.

## Core hypotheses

1. Lag-aware nonlinear models improve on persistence for sufficiently predictable variables.
2. Sequence models capture temporal interactions not represented by fixed lag features.
3. Model ranking can vary by forecast horizon and meteorological regime.
4. Point accuracy alone is insufficient; calibration and temporal stability also matter.

## Data protocol

- sort observations chronologically
- remove duplicate timestamps
- validate numeric variables
- interpolate only short gaps
- avoid interpolation across long missing periods
- record sensor/unit assumptions
- never use future data to construct a feature at time t

## Evaluation protocol

Use three complementary levels:

### 1. Holdout evaluation

Chronological train/validation/test split for fast iteration.

### 2. Rolling-origin backtesting

Repeated expanding-window evaluation:
- train on the past
- forecast a fixed future block
- move the origin forward
- aggregate fold-level performance

### 3. Regime-specific analysis

Cluster daily signatures and report errors by meteorological regime.

## Baselines

Always include:
- persistence / last observation
- a classical tree-based model
- at least one sequence model

## Metrics

Point forecasts:
- MAE
- RMSE
- R²
- signed bias

Uncertainty:
- empirical coverage
- mean interval width

Research reporting:
- fold mean and standard deviation
- bootstrap confidence intervals
- skill score relative to persistence

## Ablations

Recommended:
- no calendar features
- no rolling features
- no humidity
- no pressure
- no wind
- short vs long sequence windows
- reduced model capacity

## Forecast horizons

At minimum evaluate:
- 1 hour
- 6 hours
- 24 hours

Optional:
- 72 hours
- 168 hours

## Reproducibility checklist

- fixed random seeds
- exact package versions recorded
- model config committed
- train/validation/test periods documented
- preprocessing fit on train only
- metrics exported to CSV/JSON
- plots generated from saved predictions
- negative results retained
- all comparisons use identical folds

## Research caveats

Station-level forecasting is not a substitute for numerical weather prediction. Stronger studies should incorporate neighboring stations, radar/satellite products, topography, reanalysis, or NWP model outputs, especially for precipitation and longer lead times.
