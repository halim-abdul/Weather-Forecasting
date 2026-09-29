from __future__ import annotations

import argparse
from pathlib import Path

from weather_forecasting.classical import build_model
from weather_forecasting.data import clean_weather_frame, load_weather_csv
from weather_forecasting.evaluation import backtest_regressor, rolling_origin_folds
from weather_forecasting.features import add_calendar_features, add_lag_features, make_supervised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="temperature")
    parser.add_argument("--model", default="hist_gb")
    parser.add_argument("--initial-train", type=int, default=24 * 60)
    parser.add_argument("--horizon", type=int, default=24 * 7)
    parser.add_argument("--max-folds", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/backtest")
    args = parser.parse_args()

    frame = add_calendar_features(clean_weather_frame(load_weather_csv(args.data)))
    frame = add_lag_features(frame, [args.target, "humidity", "pressure", "wind_speed"])
    x, y = make_supervised(frame, args.target, horizon=1)
    folds = rolling_origin_folds(
        len(x), args.initial_train, args.horizon, max_folds=args.max_folds
    )
    metrics, predictions = backtest_regressor(build_model(args.model), x, y, folds)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "fold_metrics.csv", index=False)
    predictions.to_csv(out / "predictions.csv", index=False)
    print(metrics)
    print("\nMean metrics:\n", metrics.drop(columns="fold").mean())


if __name__ == "__main__":
    main()
