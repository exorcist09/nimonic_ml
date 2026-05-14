from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocess import FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMNS


DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: python src/train.py"
        )
    return joblib.load(model_path)


def predict_response(
    rake_angle: str,
    nose_radius: float,
    grain_size: str,
    cutting_speed: float,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    model = load_model(model_path)
    input_frame = pd.DataFrame(
        [
            {
                "rake_angle": str(rake_angle),
                "nose_radius": nose_radius,
                "grain_size": grain_size.lower().strip(),
                "cutting_speed": cutting_speed,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(input_frame)
    return pd.DataFrame(prediction, columns=TARGET_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict RF1, RF2, and peak temperature for Nimonic 263 end milling."
    )
    parser.add_argument("--rake-angle", default="5", help="Tool rake angle, e.g. 5 or -5.")
    parser.add_argument("--nose-radius", type=float, default=0.2, help="Tool nose radius.")
    parser.add_argument("--grain-size", default="fine", help="fine, medium, or coarse.")
    parser.add_argument("--cutting-speed", type=float, default=0.5, help="Cutting speed.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_response(
        rake_angle=args.rake_angle,
        nose_radius=args.nose_radius,
        grain_size=args.grain_size,
        cutting_speed=args.cutting_speed,
        model_path=args.model_path,
    )
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
