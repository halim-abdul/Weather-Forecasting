from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

from .classical import regression_metrics


@dataclass(frozen=True)
class BacktestFold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def rolling_origin_folds(
    n_samples: int,
    initial_train: int,
    horizon: int,
    step: int | None = None,
    max_folds: int | None = None,
) -> list[BacktestFold]:
    """Expanding-window rolling-origin evaluation folds."""
    if initial_train < 1 or horizon < 1:
        raise ValueError("initial_train and horizon must be positive")
    step = horizon if step is None else step
    folds: list[BacktestFold] = []
    train_end = initial_train
    while train_end + horizon <= n_samples:
        folds.append(BacktestFold(0, train_end, train_end, train_end + horizon))
        train_end += step
        if max_folds is not None and len(folds) >= max_folds:
            break
    return folds


def backtest_regressor(model, x: pd.DataFrame, y: pd.Series, folds: list[BacktestFold]):
    rows = []
    predictions = []
    for fold_id, fold in enumerate(folds):
        fitted = clone(model)
        x_train = x.iloc[fold.train_start:fold.train_end]
        y_train = y.iloc[fold.train_start:fold.train_end]
        x_test = x.iloc[fold.test_start:fold.test_end]
        y_test = y.iloc[fold.test_start:fold.test_end]
        fitted.fit(x_train, y_train)
        pred = fitted.predict(x_test)
        metrics = regression_metrics(y_test, pred)
        rows.append({"fold": fold_id, **metrics.__dict__})
        predictions.append(
            pd.DataFrame(
                {
                    "fold": fold_id,
                    "index": x_test.index,
                    "observed": np.asarray(y_test),
                    "forecast": np.asarray(pred),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def skill_score(model_error: float, reference_error: float) -> float:
    """Positive values indicate improvement over the reference model."""
    if reference_error <= 0:
        raise ValueError("reference_error must be positive")
    return 1.0 - model_error / reference_error


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """IID bootstrap confidence interval for a scalar metric."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        values.append(metric(y_true[idx], y_pred[idx]))
    values = np.asarray(values)
    point = metric(y_true, y_pred)
    return (
        float(point),
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )
