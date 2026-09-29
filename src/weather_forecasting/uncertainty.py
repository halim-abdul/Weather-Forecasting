from __future__ import annotations

import numpy as np


def conformal_interval(
    calibration_true,
    calibration_pred,
    test_pred,
    alpha: float = 0.1,
):
    """Distribution-free split-conformal symmetric prediction intervals."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    cal_true = np.asarray(calibration_true)
    cal_pred = np.asarray(calibration_pred)
    test_pred = np.asarray(test_pred)
    scores = np.abs(cal_true - cal_pred)
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q = np.quantile(scores, q_level, method="higher")
    return test_pred - q, test_pred + q


def interval_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
