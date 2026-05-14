from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from evaluate import regression_metrics, save_metrics, summarize_average_metrics
from preprocess import (
    PROJECT_ROOT,
    RAW_DATA_PATH,
    build_preprocessor,
    clean_dataset,
    configure_logging,
    load_dataset,
    save_processed_dataset,
    split_features_targets,
)
from visualize import (
    save_correlation_heatmap,
    save_feature_importance_plot,
    save_pair_plot,
    save_prediction_plots,
    save_residual_plot,
)


MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
LOGGER = logging.getLogger(__name__)


def build_models() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        ),
        "svr": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.05))),
            ]
        ),
        "mlp": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        max_iter=2000,
                        random_state=42,
                        early_stopping=False,
                    ),
                ),
            ]
        ),
    }


def cross_validate_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    splits = min(5, len(x_train))
    if splits < 2:
        return {"n_splits": splits, "negative_mae_scores": [], "mean_mae": None}

    cv = KFold(n_splits=splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        x_train,
        y_train,
        scoring="neg_mean_absolute_error",
        cv=cv,
    )
    return {
        "n_splits": splits,
        "negative_mae_scores": [float(score) for score in scores],
        "mean_mae": float(-np.mean(scores)),
    }


def train_all_models(data_path: Path = RAW_DATA_PATH) -> dict:
    configure_logging()
    raw_data = load_dataset(data_path)
    cleaned_data = clean_dataset(raw_data)
    x, y = split_features_targets(cleaned_data)

    if len(cleaned_data) < 5:
        LOGGER.warning(
            "Dataset has only %s rows. Metrics will be unstable; add more simulations when available.",
            len(cleaned_data),
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    metrics: dict[str, dict] = {}
    trained_models: dict[str, Pipeline] = {}
    for model_name, model in build_models().items():
        LOGGER.info("Training %s model.", model_name)
        cv_metrics = cross_validate_model(model, x_train, y_train)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        target_metrics = regression_metrics(y_test, predictions)
        metrics[model_name] = {
            "cross_validation": cv_metrics,
            "test_metrics_by_target": target_metrics,
            "test_metrics_average": summarize_average_metrics(target_metrics),
        }
        trained_models[model_name] = model

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    random_forest_model = trained_models["random_forest"]
    joblib.dump(random_forest_model, DEFAULT_MODEL_PATH)
    LOGGER.info("Saved default Random Forest surrogate model to %s", DEFAULT_MODEL_PATH)

    random_forest_predictions = random_forest_model.predict(x_test)
    save_processed_dataset(data_path)
    save_metrics(metrics)
    save_correlation_heatmap(cleaned_data)
    save_pair_plot(cleaned_data)
    save_prediction_plots(y_test, random_forest_predictions)
    save_residual_plot(y_test, random_forest_predictions)
    save_feature_importance_plot(random_forest_model)

    return metrics


if __name__ == "__main__":
    train_all_models()
