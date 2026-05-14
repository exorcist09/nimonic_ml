from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from preprocess import FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMNS


PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def _prepare_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300})


def save_correlation_heatmap(data: pd.DataFrame) -> Path:
    _prepare_plots_dir()
    numeric_data = pd.get_dummies(data[FEATURE_COLUMNS + TARGET_COLUMNS], drop_first=False)
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    output_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_pair_plot(data: pd.DataFrame) -> Path:
    _prepare_plots_dir()
    plot = sns.pairplot(data[FEATURE_COLUMNS + TARGET_COLUMNS], hue="grain_size", diag_kind="hist")
    plot.fig.suptitle("Machining Dataset Pair Plot", y=1.02)
    output_path = PLOTS_DIR / "pair_plot.png"
    plot.savefig(output_path)
    plt.close(plot.fig)
    return output_path


def save_prediction_plots(y_true: pd.DataFrame, y_pred: np.ndarray) -> list[Path]:
    _prepare_plots_dir()
    paths: list[Path] = []
    for index, target in enumerate(TARGET_COLUMNS):
        actual = y_true.iloc[:, index]
        predicted = y_pred[:, index]
        plt.figure(figsize=(5.5, 4.5))
        sns.scatterplot(x=actual, y=predicted, s=55, label="Samples")
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        plt.plot(
            [low, high],
            [low, high],
            color="black",
            linestyle="--",
            linewidth=1,
            label="Ideal prediction (y=x)",
        )
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.title(f"{target} Prediction vs Actual")
        plt.legend(loc="best")
        plt.tight_layout()
        output_path = PLOTS_DIR / f"{target.lower()}_prediction_vs_actual.png"
        plt.savefig(output_path)
        plt.close()
        paths.append(output_path)
    return paths


def save_residual_plot(y_true: pd.DataFrame, y_pred: np.ndarray) -> Path:
    _prepare_plots_dir()
    residuals = y_true.to_numpy() - y_pred
    residual_frame = pd.DataFrame(residuals, columns=TARGET_COLUMNS)
    long_residuals = residual_frame.melt(var_name="target", value_name="residual")
    plt.figure(figsize=(7, 4.5))
    sns.boxplot(data=long_residuals, x="target", y="residual")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Target")
    plt.ylabel("Actual - Predicted")
    plt.title("Residual Error Plot")
    legend_handles = [
        Patch(facecolor="#4C72B0", edgecolor="#4C72B0", label="Residual distribution"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Zero error"),
    ]
    plt.legend(handles=legend_handles, loc="best")
    plt.tight_layout()
    output_path = PLOTS_DIR / "residual_error_plot.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def save_feature_importance_plot(model_pipeline) -> Path | None:
    _prepare_plots_dir()
    model = model_pipeline.named_steps.get("model")
    if not hasattr(model, "feature_importances_"):
        return None

    preprocessor = model_pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=importance, x="importance", y="feature", color="#4C78A8")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.legend(
        handles=[Patch(facecolor="#4C78A8", edgecolor="#4C78A8", label="Feature importance")],
        loc="best",
    )
    plt.tight_layout()
    output_path = PLOTS_DIR / "feature_importance.png"
    plt.savefig(output_path)
    plt.close()
    return output_path
