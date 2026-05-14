from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "machining_data.xlsx"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

FEATURE_COLUMNS = ["rake_angle", "nose_radius", "grain_size", "cutting_speed"]
TARGET_COLUMNS = ["RF1", "RF2", "temperature"]
CATEGORICAL_COLUMNS = ["rake_angle", "grain_size"]
NUMERIC_COLUMNS = ["nose_radius", "cutting_speed"]

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load machining data from an Excel workbook."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place your Abaqus/experimental workbook "
            "at data/raw/machining_data.xlsx."
        )
    return pd.read_excel(path, engine="openpyxl")


def validate_columns(
    data: pd.DataFrame,
    features: Iterable[str] = FEATURE_COLUMNS,
    targets: Iterable[str] = TARGET_COLUMNS,
) -> None:
    required = list(features) + list(targets)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing model inputs or targets and normalize categories."""
    validate_columns(data)
    selected = data[FEATURE_COLUMNS + TARGET_COLUMNS].copy()

    missing_rows = int(selected.isna().any(axis=1).sum())
    if missing_rows:
        LOGGER.warning("Dropping %s rows with missing feature/target values.", missing_rows)
        selected = selected.dropna(axis=0)

    selected["rake_angle"] = selected["rake_angle"].astype(str).str.strip()
    selected["grain_size"] = selected["grain_size"].astype(str).str.strip().str.lower()

    for column in NUMERIC_COLUMNS + TARGET_COLUMNS:
        selected[column] = pd.to_numeric(selected[column], errors="raise")

    return selected.reset_index(drop=True)


def split_features_targets(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = clean_dataset(data)
    return cleaned[FEATURE_COLUMNS], cleaned[TARGET_COLUMNS]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), CATEGORICAL_COLUMNS),
            ("numeric", MinMaxScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def save_processed_dataset(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    """Create a transparent processed CSV artifact for inspection/reporting."""
    configure_logging()
    raw = load_dataset(input_path)
    cleaned = clean_dataset(raw)
    preprocessor = build_preprocessor()
    encoded = preprocessor.fit_transform(cleaned[FEATURE_COLUMNS])
    feature_names = preprocessor.get_feature_names_out()
    processed = pd.DataFrame(encoded, columns=feature_names)
    processed[TARGET_COLUMNS] = cleaned[TARGET_COLUMNS].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    LOGGER.info("Saved processed dataset to %s", output_path)
    return output_path


if __name__ == "__main__":
    save_processed_dataset()
