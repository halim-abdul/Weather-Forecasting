from __future__ import annotations

import argparse
import json
from pathlib import Path

from weather_forecasting.classical import build_model, fit_and_evaluate
from weather_forecasting.data import SplitConfig, chronological_split, clean_weather_frame, load_weather_csv
from weather_forecasting.features import add_calendar_features, add_lag_features, make_supervised
from weather_forecasting.visualization import save_forecast_plot, save_residual_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="temperature")
    parser.add_argument("--model", default="hist_gb", choices=["hist_gb", "random_forest", "extra_trees"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-dir", default="artifacts/sklearn")
    args = parser.parse_args()

    frame = clean_weather_frame(load_weather_csv(args.data))
    frame = add_calendar_features(frame)
    frame = add_lag_features(frame, [args.target, "humidity", "pressure", "wind_speed"])
    x, y = make_supervised(frame, args.target, args.horizon)

    supervised = x.copy()
    supervised["_target"] = y
    train, _, test = chronological_split(supervised, SplitConfig())
    feature_cols = [c for c in supervised.columns if c != "_target"]

    model = build_model(args.model)
    model, pred, metrics = fit_and_evaluate(
        model,
        train[feature_cols],
        train["_target"],
        test[feature_cols],
        test["_target"],
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics.__dict__, indent=2))
    save_forecast_plot(test.index, test["_target"], pred, out / "forecast.png")
    save_residual_diagnostics(test["_target"], pred, out / "residuals.png")
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
