from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_forecast_plot(
    timestamps,
    y_true,
    y_pred,
    path: str | Path,
    title: str = "Weather forecast vs observation",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(timestamps, y_true, label="Observed", linewidth=1.6)
    ax.plot(timestamps, y_pred, label="Forecast", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, path)


def save_residual_diagnostics(y_true, y_pred, path: str | Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=14, alpha=0.55)
    ax.axhline(0, linewidth=1)
    ax.set_title("Residual diagnostics")
    ax.set_xlabel("Forecast")
    ax.set_ylabel("Observed - forecast")
    ax.grid(alpha=0.2)
    _save(fig, path)


def save_correlation_heatmap(frame: pd.DataFrame, path: str | Path) -> None:
    corr = frame.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    _save(fig, path)


def save_regime_embedding(regimes: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        regimes["pc1"],
        regimes["pc2"],
        c=regimes["regime"],
        cmap="tab10",
        s=24,
        alpha=0.75,
    )
    ax.set_title("Weather-regime embedding")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Regime")
    _save(fig, path)


def save_error_by_hour(
    timestamps,
    y_true,
    y_pred,
    path: str | Path,
) -> None:
    ts = pd.to_datetime(timestamps, utc=True)
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    table = pd.DataFrame({"hour": ts.hour, "absolute_error": errors})
    summary = table.groupby("hour")["absolute_error"].mean().reindex(range(24))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(summary.index, summary.values)
    ax.set_title("Mean absolute forecast error by hour of day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def save_interval_plot(
    timestamps,
    y_true,
    y_pred,
    lower,
    upper,
    path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(timestamps, lower, upper, alpha=0.25, label="Prediction interval")
    ax.plot(timestamps, y_true, linewidth=1.4, label="Observed")
    ax.plot(timestamps, y_pred, linewidth=1.2, label="Forecast")
    ax.set_title("Forecast uncertainty")
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(alpha=0.2)
    _save(fig, path)
