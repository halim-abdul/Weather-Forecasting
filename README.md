# Weather-Forecasting

**Big Data Analytics for Weather Pattern Detection and Forecasting with AI**

A compact, research-grade Python project for weather time-series analysis, pattern discovery, classical machine learning, deep learning, uncertainty quantification, robust backtesting, and publication-ready diagnostics.

## What is included

- **Data engineering**
  - timestamp parsing and validation
  - duplicate removal
  - interpolation for short missing gaps
  - robust MAD-based clipping
  - chronological train/validation/test splitting

- **Feature engineering**
  - cyclical hour-of-day and day-of-year encoding
  - lagged weather variables
  - past-only rolling statistics
  - leakage-safe supervised targets

- **scikit-learn forecasting**
  - HistGradientBoosting
  - Random Forest
  - Extra Trees
  - MAE, RMSE, R² and forecast bias

- **PyTorch sequence models**
  - LSTM
  - GRU
  - dilated temporal CNN
  - AdamW
  - SmoothL1 loss
  - gradient clipping
  - ReduceLROnPlateau
  - early stopping
  - GPU support

- **Research evaluation**
  - expanding-window rolling-origin backtesting
  - persistence/reference-model comparison
  - forecast skill scores
  - bootstrap confidence intervals
  - split-conformal prediction intervals
  - coverage and interval-width diagnostics

- **Weather pattern discovery**
  - daily meteorological signatures
  - PCA embeddings
  - KMeans regime clustering

- **Visualization**
  - observed vs forecast trajectories
  - residual diagnostics
  - correlation maps
  - notebook-based exploratory analysis

## Repository structure

```text
Weather-Forecasting/
├── configs/
│   └── default.yaml
├── docs/
│   └── research_protocol.md
├── notebooks/
│   └── 01_weather_research_workbench.ipynb
├── scripts/
│   ├── backtest.py
│   ├── generate_synthetic_weather.py
│   ├── train_sklearn.py
│   └── train_torch.py
├── src/weather_forecasting/
│   ├── classical.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── patterns.py
│   ├── torch_data.py
│   ├── torch_models.py
│   ├── training.py
│   ├── uncertainty.py
│   └── visualization.py
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

## Expected CSV schema

At minimum:

```text
timestamp,temperature,humidity,pressure,wind_speed,precipitation
```

The code assumes a regularly sampled weather time series, ideally hourly. Additional numeric variables can be retained and used as covariates.

## Quick start

```bash
git clone https://github.com/halim-abdul/Weather-Forecasting.git
cd Weather-Forecasting

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -e ".[deep,notebook,dev]"
```

Generate a self-contained synthetic weather dataset:

```bash
python scripts/generate_synthetic_weather.py --output data/weather.csv --days 730
```

Train a classical model:

```bash
python scripts/train_sklearn.py \
  --data data/weather.csv \
  --target temperature \
  --model hist_gb \
  --horizon 1
```

Run rolling-origin research backtesting:

```bash
python scripts/backtest.py \
  --data data/weather.csv \
  --target temperature \
  --model hist_gb \
  --initial-train 8760 \
  --horizon 168 \
  --max-folds 8
```

Train a PyTorch LSTM:

```bash
python scripts/train_torch.py \
  --data data/weather.csv \
  --target temperature \
  --model lstm \
  --sequence-length 48 \
  --horizon 1 \
  --epochs 40
```

## Experimental design

For defensible time-series experiments:

1. Never randomly shuffle timestamps before evaluation.
2. Fit preprocessing only on historical training data.
3. Compare every AI model with a persistence baseline.
4. Report multiple metrics instead of only R².
5. Evaluate multiple forecast horizons.
6. Use rolling-origin backtesting to measure stability over time.
7. Report uncertainty on both forecasts and aggregate metrics.
8. Preserve seeds, configs and environment information.

See [docs/research_protocol.md](docs/research_protocol.md).

## Suggested research experiments

| Experiment | Question |
|---|---|
| Persistence vs tree ensembles | Do nonlinear lag interactions outperform naive forecasting? |
| LSTM vs GRU vs TCN | Which temporal inductive bias works best for hourly weather? |
| 1h / 6h / 24h horizons | How quickly does predictability degrade with lead time? |
| Seasonal subsets | Does performance vary between winter and summer? |
| Regime-aware evaluation | Which weather regimes are hardest to forecast? |
| Feature ablation | How much value comes from pressure, humidity and wind? |
| Conformal intervals | Are empirical forecast intervals correctly calibrated? |
| Rolling-origin folds | Is performance stable through time or dependent on one split? |

## Reproducibility

The repository includes:
- explicit random seeds
- chronological data partitioning
- unit tests
- configurable training
- GitHub Actions CI
- artifact directories for metrics, histories, plots and predictions

Run tests with:

```bash
pytest -q
```

## Notes on scope

This repository is designed as a research/portfolio forecasting framework. For operational numerical weather prediction, one would additionally need spatial gridded fields, physically informed features, atmospheric reanalysis or NWP inputs, multi-location modeling, and substantially larger compute/data infrastructure.

## License

Add the license appropriate for your intended reuse before external redistribution.
