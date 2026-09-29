import numpy as np
import pandas as pd

from weather_forecasting.data import SplitConfig, chronological_split
from weather_forecasting.evaluation import rolling_origin_folds
from weather_forecasting.features import add_calendar_features, add_lag_features, make_supervised
from weather_forecasting.uncertainty import conformal_interval, interval_coverage


def _frame(n=100):
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temperature": np.linspace(5, 15, n),
            "humidity": np.linspace(70, 50, n),
            "pressure": 1010 + np.sin(np.arange(n) / 10),
            "wind_speed": 3 + np.cos(np.arange(n) / 5),
            "precipitation": np.zeros(n),
        }
    )


def test_lag_features_use_past_values():
    df = add_lag_features(_frame(), ["temperature"], lags=(1,), rolling_windows=(3,))
    assert df.loc[10, "temperature_lag_1"] == df.loc[9, "temperature"]


def test_supervised_horizon_alignment():
    df = add_calendar_features(_frame())
    x, y = make_supervised(df, target="temperature", horizon=2)
    first_idx = x.index[0]
    assert y.loc[first_idx] == df.loc[first_idx + 2, "temperature"]


def test_chronological_split_order():
    a, b, c = chronological_split(_frame(), SplitConfig(0.6, 0.2))
    assert a.index.max() < b.index.min() < c.index.min()


def test_rolling_origin_is_forward_only():
    folds = rolling_origin_folds(100, initial_train=50, horizon=10)
    assert all(f.train_end <= f.test_start for f in folds)


def test_conformal_interval_shapes_and_coverage():
    true = np.array([1, 2, 3, 4, 5])
    pred = np.array([1.1, 1.8, 3.2, 4.1, 4.8])
    lower, upper = conformal_interval(true, pred, np.array([2.5, 3.5]), alpha=0.2)
    assert lower.shape == upper.shape == (2,)
    assert interval_coverage(np.array([2.5, 3.5]), lower, upper) == 1.0
