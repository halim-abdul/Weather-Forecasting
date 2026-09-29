from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_weather(days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24
    t = np.arange(n)
    timestamp = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    annual = 10 * np.sin(2 * np.pi * t / (365.25 * 24) - 1.0)
    daily = 4 * np.sin(2 * np.pi * t / 24 - 0.8)
    synoptic = 2.5 * np.sin(2 * np.pi * t / (24 * 7))
    temperature = 11 + annual + daily + synoptic + rng.normal(0, 1.3, n)

    humidity = 72 - 0.9 * daily - 0.55 * annual + rng.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 100)

    pressure = (
        1014
        + 6 * np.sin(2 * np.pi * t / (24 * 9) + 0.6)
        + 3 * np.sin(2 * np.pi * t / (24 * 4.5))
        + rng.normal(0, 1.5, n)
    )

    wind_speed = np.clip(
        3.5
        + 1.4 * np.sin(2 * np.pi * t / (24 * 3.2))
        + rng.gamma(shape=1.4, scale=0.7, size=n),
        0,
        None,
    )

    rain_probability = 1 / (1 + np.exp((pressure - 1009) / 2.5))
    wet = rng.random(n) < 0.12 * rain_probability
    precipitation = np.where(wet, rng.gamma(1.6, 1.8, n), 0.0)

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "temperature": temperature,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "precipitation": precipitation,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/weather.csv")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = generate_weather(args.days, args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame):,} rows to {path}")


if __name__ == "__main__":
    main()
