from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from weather_forecasting.classical import regression_metrics
from weather_forecasting.data import clean_weather_frame, load_weather_csv
from weather_forecasting.features import add_calendar_features
from weather_forecasting.torch_data import SequenceDataset
from weather_forecasting.torch_models import GRUForecaster, LSTMForecaster, TemporalConvForecaster
from weather_forecasting.training import fit_model, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="temperature")
    parser.add_argument("--model", choices=["lstm", "gru", "tcn"], default="lstm")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output-dir", default="artifacts/pytorch")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    frame = add_calendar_features(clean_weather_frame(load_weather_csv(args.data)))
    feature_cols = [
        c for c in frame.select_dtypes(include="number").columns if c != args.target
    ] + [args.target]

    n = len(frame)
    i, j = int(0.70 * n), int(0.85 * n)
    scaler = StandardScaler().fit(frame.loc[: i - 1, feature_cols])
    x = scaler.transform(frame[feature_cols]).astype("float32")
    target_idx = feature_cols.index(args.target)
    y = x[:, target_idx]

    train_ds = SequenceDataset(x[:i], y[:i], args.sequence_length, args.horizon)
    val_start = max(0, i - args.sequence_length)
    val_ds = SequenceDataset(x[val_start:j], y[val_start:j], args.sequence_length, args.horizon)
    test_start = max(0, j - args.sequence_length)
    test_ds = SequenceDataset(x[test_start:], y[test_start:], args.sequence_length, args.horizon)

    loaders = [
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size),
        DataLoader(test_ds, batch_size=args.batch_size),
    ]
    model_cls = {"lstm": LSTMForecaster, "gru": GRUForecaster, "tcn": TemporalConvForecaster}[args.model]
    model = model_cls(len(feature_cols), horizon=args.horizon)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = fit_model(model, loaders[0], loaders[1], epochs=args.epochs, device=device)
    pred_scaled, true_scaled = predict(model, loaders[2], device=device)

    scale = scaler.scale_[target_idx]
    mean = scaler.mean_[target_idx]
    pred = pred_scaled[:, 0] * scale + mean
    truth = true_scaled[:, 0] * scale + mean
    metrics = regression_metrics(truth, pred)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "features": feature_cols}, out / "model.pt")
    pd.DataFrame([s.__dict__ for s in history]).to_csv(out / "history.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics.__dict__, indent=2))
    print(json.dumps(metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
