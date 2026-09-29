from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_WEATHER_COLUMNS = (
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "precipitation",
)


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.70
    val_fraction: float = 0.15

    def validate(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must be < 1")


def load_weather_csv(
    path: str | Path,
    timestamp_col: str = "timestamp",
    required_columns: Iterable[str] = DEFAULT_WEATHER_COLUMNS,
) -> pd.DataFrame:
    """Load, validate, sort and de-duplicate a weather time-series CSV."""
    frame = pd.read_csv(path)
    required = {timestamp_col, *required_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
    if frame[timestamp_col].isna().any():
        raise ValueError("Invalid timestamps found")

    frame = frame.sort_values(timestamp_col).drop_duplicates(timestamp_col).reset_index(drop=True)
    numeric = list(required.difference({timestamp_col}))
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame


def clean_weather_frame(frame: pd.DataFrame, max_gap: int = 6) -> pd.DataFrame:
    """Clean numeric columns with conservative interpolation and robust clipping."""
    out = frame.copy()
    numeric = out.select_dtypes(include=np.number).columns
    out[numeric] = out[numeric].interpolate(limit=max_gap, limit_direction="both")

    for col in numeric:
        median = out[col].median()
        mad = np.median(np.abs(out[col] - median))
        if mad > 0:
            lo, hi = median - 8 * 1.4826 * mad, median + 8 * 1.4826 * mad
            out[col] = out[col].clip(lo, hi)

    return out.dropna().reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, config: SplitConfig = SplitConfig()
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Leakage-safe chronological train/validation/test split."""
    config.validate()
    n = len(frame)
    i = int(n * config.train_fraction)
    j = int(n * (config.train_fraction + config.val_fraction))
    return frame.iloc[:i].copy(), frame.iloc[i:j].copy(), frame.iloc[j:].copy()
