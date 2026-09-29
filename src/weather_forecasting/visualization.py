from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_forecast_plot(
    timestamps,
    y_true,
    y_pred,
    path: str | Path,
    title: str = "Weather forecast vs observation",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(timestamps, y_true, label="Observed", linewidth=1.6)
    ax.plot(timestamps, y_pred, label="Forecast", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_residual_diagnostics(y_true, y_pred, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=14, alpha=0.55)
    ax.axhline(0, linewidth=1)
    ax.set_title("Residual diagnostics")
    ax.set_xlabel("Forecast")
    ax.set_ylabel("Observed - forecast")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_correlation_heatmap(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corr = frame.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
