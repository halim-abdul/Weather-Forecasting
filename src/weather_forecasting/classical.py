from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    bias: float


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
        bias=float(np.mean(y_pred - y_true)),
    )


def build_model(name: str = "hist_gb", random_state: int = 42):
    models = {
        "hist_gb": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1e-3,
            random_state=random_state,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(models)}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", models[name])])


def persistence_forecast(frame: pd.DataFrame, target: str = "temperature") -> np.ndarray:
    return frame[target].shift(1).to_numpy()


def fit_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, pred, regression_metrics(y_test, pred)
