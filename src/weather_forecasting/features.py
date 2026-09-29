from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)
    hour = ts.dt.hour.to_numpy()
    doy = ts.dt.dayofyear.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_lag_features(
    frame: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    rolling_windows: tuple[int, ...] = (3, 6, 24),
) -> pd.DataFrame:
    """Create past-only lag and rolling statistics to avoid target leakage."""
    out = frame.copy()
    for col in columns:
        shifted = out[col].shift(1)
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for window in rolling_windows:
            roll = shifted.rolling(window)
            out[f"{col}_roll_mean_{window}"] = roll.mean()
            out[f"{col}_roll_std_{window}"] = roll.std()
    return out


def make_supervised(
    frame: pd.DataFrame,
    target: str = "temperature",
    horizon: int = 1,
    drop_columns: tuple[str, ...] = ("timestamp",),
) -> tuple[pd.DataFrame, pd.Series]:
    out = frame.copy()
    y = out[target].shift(-horizon).rename(f"{target}_t_plus_{horizon}")
    x = out.drop(columns=list(drop_columns), errors="ignore")
    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]
