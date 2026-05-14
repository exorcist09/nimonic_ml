from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import PROJECT_ROOT, TARGET_COLUMNS


METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def regression_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Calculate MAE, RMSE, and R2 for each machining response."""
    results: dict[str, dict[str, float]] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        actual = y_true.iloc[:, index]
        predicted = y_pred[:, index]
        results[target] = {
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted)),
        }
    return results


def summarize_average_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: float(np.mean([target_metrics[metric] for target_metrics in metrics.values()]))
        for metric in ["MAE", "RMSE", "R2"]
    }


def save_metrics(metrics: dict, filename: str = "model_metrics.json") -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / filename
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return output_path
